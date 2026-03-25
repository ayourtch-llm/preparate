use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Seek, Write};
use std::path::Path;

use memmap2::Mmap;

// ── GGUF magic & alignment ──────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32
const DEFAULT_ALIGNMENT: u64 = 32;

// ── GgmlType ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 are removed (Q4_2, Q4_3)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    // TQ1_0 = 34,
    // TQ2_0 = 35,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ1S => "IQ1_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ3S => "IQ3_S",
            Self::IQ2S => "IQ2_S",
            Self::IQ4XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1M => "IQ1_M",
            Self::BF16 => "BF16",
        }
    }

    /// (block_size_in_elements, bytes_per_block)
    pub fn block_info(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 40),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::IQ2XXS => (256, 66),
            Self::IQ2XS => (256, 74),
            Self::IQ3XXS => (256, 98),
            Self::IQ1S => (256, 50),
            Self::IQ4NL => (32, 18),
            Self::IQ3S => (256, 110),
            Self::IQ2S => (256, 82),
            Self::IQ4XS => (256, 136),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::F64 => (1, 8),
            Self::IQ1M => (256, 56),
            Self::BF16 => (1, 2),
        }
    }
}

// ── Metadata value types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

// ── Parsed metadata values ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        elem_type: MetadataValueType,
        values: Vec<MetadataValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn value_type(&self) -> MetadataValueType {
        match self {
            Self::Uint8(_) => MetadataValueType::Uint8,
            Self::Int8(_) => MetadataValueType::Int8,
            Self::Uint16(_) => MetadataValueType::Uint16,
            Self::Int16(_) => MetadataValueType::Int16,
            Self::Uint32(_) => MetadataValueType::Uint32,
            Self::Int32(_) => MetadataValueType::Int32,
            Self::Float32(_) => MetadataValueType::Float32,
            Self::Bool(_) => MetadataValueType::Bool,
            Self::String(_) => MetadataValueType::String,
            Self::Array { .. } => MetadataValueType::Array,
            Self::Uint64(_) => MetadataValueType::Uint64,
            Self::Int64(_) => MetadataValueType::Int64,
            Self::Float64(_) => MetadataValueType::Float64,
        }
    }

    pub fn display_short(&self) -> String {
        match self {
            Self::Uint8(v) => format!("{v}"),
            Self::Int8(v) => format!("{v}"),
            Self::Uint16(v) => format!("{v}"),
            Self::Int16(v) => format!("{v}"),
            Self::Uint32(v) => format!("{v}"),
            Self::Int32(v) => format!("{v}"),
            Self::Float32(v) => format!("{v}"),
            Self::Bool(v) => format!("{v}"),
            Self::String(v) => {
                if v.len() > 80 {
                    format!("\"{}...\" ({} bytes)", &v[..77], v.len())
                } else {
                    format!("\"{v}\"")
                }
            }
            Self::Array { values, elem_type } => {
                format!("[{:?}; {} elements]", elem_type, values.len())
            }
            Self::Uint64(v) => format!("{v}"),
            Self::Int64(v) => format!("{v}"),
            Self::Float64(v) => format!("{v}"),
        }
    }
}

// ── Tensor info ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub typ: GgmlType,
    pub offset: u64, // relative to tensor data start
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dimensions.iter().copied().product::<u64>().max(1)
    }

    pub fn data_size(&self) -> u64 {
        let (block_size, bytes_per_block) = self.typ.block_info();
        let n = self.n_elements();
        let n_blocks = (n + block_size - 1) / block_size;
        n_blocks * bytes_per_block
    }

    /// Returns Some(layer_number) if this tensor belongs to a transformer block.
    pub fn block_number(&self) -> Option<u32> {
        if self.name.starts_with("blk.") {
            let rest = &self.name[4..];
            if let Some(dot) = rest.find('.') {
                rest[..dot].parse().ok()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Returns the part after `blk.N.` if this is a block tensor.
    pub fn block_suffix(&self) -> Option<&str> {
        if self.name.starts_with("blk.") {
            let rest = &self.name[4..];
            rest.find('.').map(|dot| &rest[dot + 1..])
        } else {
            None
        }
    }
}

// ── Binary reader helper ────────────────────────────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "need {} bytes at offset {}, but file has {} bytes",
                    n,
                    self.pos,
                    self.data.len()
                ),
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> io::Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> io::Result<i16> {
        Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn read_bool(&mut self) -> io::Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_metadata_value(&mut self, vtype: MetadataValueType) -> io::Result<MetadataValue> {
        match vtype {
            MetadataValueType::Uint8 => Ok(MetadataValue::Uint8(self.read_u8()?)),
            MetadataValueType::Int8 => Ok(MetadataValue::Int8(self.read_i8()?)),
            MetadataValueType::Uint16 => Ok(MetadataValue::Uint16(self.read_u16()?)),
            MetadataValueType::Int16 => Ok(MetadataValue::Int16(self.read_i16()?)),
            MetadataValueType::Uint32 => Ok(MetadataValue::Uint32(self.read_u32()?)),
            MetadataValueType::Int32 => Ok(MetadataValue::Int32(self.read_i32()?)),
            MetadataValueType::Float32 => Ok(MetadataValue::Float32(self.read_f32()?)),
            MetadataValueType::Bool => Ok(MetadataValue::Bool(self.read_bool()?)),
            MetadataValueType::String => Ok(MetadataValue::String(self.read_string()?)),
            MetadataValueType::Array => {
                let elem_type_raw = self.read_u32()?;
                let elem_type = MetadataValueType::from_u32(elem_type_raw).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown array element type: {elem_type_raw}"),
                    )
                })?;
                let count = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(count.min(1_000_000));
                for _ in 0..count {
                    values.push(self.read_metadata_value(elem_type)?);
                }
                Ok(MetadataValue::Array { elem_type, values })
            }
            MetadataValueType::Uint64 => Ok(MetadataValue::Uint64(self.read_u64()?)),
            MetadataValueType::Int64 => Ok(MetadataValue::Int64(self.read_i64()?)),
            MetadataValueType::Float64 => Ok(MetadataValue::Float64(self.read_f64()?)),
        }
    }
}

// ── GgufFile ────────────────────────────────────────────────────────────────

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: Vec<(String, MetadataValue)>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64,
    pub alignment: u64,
}

impl GgufFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cur = Cursor::new(&mmap);

        // Read header
        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("not a GGUF file (magic: 0x{magic:08x})"),
            ));
        }

        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version} (only v2/v3 supported)"),
            ));
        }

        let tensor_count = cur.read_u64()? as usize;
        let metadata_kv_count = cur.read_u64()? as usize;

        // Read metadata
        let mut metadata = Vec::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let vtype_raw = cur.read_u32()?;
            let vtype = MetadataValueType::from_u32(vtype_raw).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown metadata value type: {vtype_raw}"),
                )
            })?;
            let value = cur.read_metadata_value(vtype)?;
            metadata.push((key, value));
        }

        // Determine alignment
        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| match v {
                MetadataValue::Uint32(a) => Some(*a as u64),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);

        // Read tensor infos
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(cur.read_u64()?);
            }
            let type_raw = cur.read_u32()?;
            let typ = GgmlType::from_u32(type_raw).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown tensor type {type_raw} for tensor '{name}'"),
                )
            })?;
            let offset = cur.read_u64()?;
            tensors.push(TensorInfo {
                name,
                dimensions,
                typ,
                offset,
            });
        }

        // Tensor data starts at next alignment boundary after current position
        let tensor_data_offset = align_offset(cur.pos as u64, alignment);

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            tensor_data_offset,
            alignment,
        })
    }

    /// Get the raw bytes for a tensor's data.
    pub fn tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        let start = self.tensor_data_offset + tensor.offset;
        let size = tensor.data_size();
        &self.mmap[start as usize..(start + size) as usize]
    }

    /// Get a metadata value by key.
    #[allow(dead_code)]
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Collect block tensors grouped by block number.
    pub fn block_layers(&self) -> BTreeMap<u32, Vec<usize>> {
        let mut map = BTreeMap::new();
        for (i, t) in self.tensors.iter().enumerate() {
            if let Some(blk) = t.block_number() {
                map.entry(blk).or_insert_with(Vec::new).push(i);
            }
        }
        map
    }

    /// Indices of non-block tensors.
    pub fn non_block_tensor_indices(&self) -> Vec<usize> {
        self.tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.block_number().is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// The total number of transformer blocks found.
    pub fn block_count(&self) -> u32 {
        self.block_layers().keys().copied().max().map_or(0, |m| m + 1)
    }
}

// ── GGUF writer ─────────────────────────────────────────────────────────────

fn align_offset(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) / alignment * alignment
}

struct GgufWriter<W: Write + Seek> {
    w: W,
    alignment: u64,
}

impl<W: Write + Seek> GgufWriter<W> {
    fn new(w: W, alignment: u64) -> Self {
        Self { w, alignment }
    }

    fn write_u8(&mut self, v: u8) -> io::Result<()> {
        self.w.write_all(&[v])
    }

    fn write_i8(&mut self, v: i8) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_u16(&mut self, v: u16) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_i16(&mut self, v: i16) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_u32(&mut self, v: u32) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_i32(&mut self, v: i32) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_u64(&mut self, v: u64) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_i64(&mut self, v: i64) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_f32(&mut self, v: f32) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_f64(&mut self, v: f64) -> io::Result<()> {
        self.w.write_all(&v.to_le_bytes())
    }

    fn write_string(&mut self, s: &str) -> io::Result<()> {
        self.write_u64(s.len() as u64)?;
        self.w.write_all(s.as_bytes())
    }

    fn write_metadata_value(&mut self, val: &MetadataValue) -> io::Result<()> {
        match val {
            MetadataValue::Uint8(v) => self.write_u8(*v),
            MetadataValue::Int8(v) => self.write_i8(*v),
            MetadataValue::Uint16(v) => self.write_u16(*v),
            MetadataValue::Int16(v) => self.write_i16(*v),
            MetadataValue::Uint32(v) => self.write_u32(*v),
            MetadataValue::Int32(v) => self.write_i32(*v),
            MetadataValue::Float32(v) => self.write_f32(*v),
            MetadataValue::Bool(v) => self.write_u8(if *v { 1 } else { 0 }),
            MetadataValue::String(v) => self.write_string(v),
            MetadataValue::Array { elem_type, values } => {
                self.write_u32(*elem_type as u32)?;
                self.write_u64(values.len() as u64)?;
                for v in values {
                    self.write_metadata_value(v)?;
                }
                Ok(())
            }
            MetadataValue::Uint64(v) => self.write_u64(*v),
            MetadataValue::Int64(v) => self.write_i64(*v),
            MetadataValue::Float64(v) => self.write_f64(*v),
        }
    }

    fn write_metadata_kv(&mut self, key: &str, val: &MetadataValue) -> io::Result<()> {
        self.write_string(key)?;
        self.write_u32(val.value_type() as u32)?;
        self.write_metadata_value(val)
    }

    fn write_tensor_info(&mut self, name: &str, info: &TensorInfo, offset: u64) -> io::Result<()> {
        self.write_string(name)?;
        self.write_u32(info.dimensions.len() as u32)?;
        for &d in &info.dimensions {
            self.write_u64(d)?;
        }
        self.write_u32(info.typ as u32)?;
        self.write_u64(offset)
    }

    fn pad_to_alignment(&mut self) -> io::Result<()> {
        let pos = self.w.stream_position()?;
        let aligned = align_offset(pos, self.alignment);
        let padding = aligned - pos;
        if padding > 0 {
            let zeros = vec![0u8; padding as usize];
            self.w.write_all(&zeros)?;
        }
        Ok(())
    }

    fn current_pos(&mut self) -> io::Result<u64> {
        self.w.stream_position()
    }
}

/// Describes a tensor to write: its (possibly renamed) name, original info, and data bytes.
pub struct OutputTensor<'a> {
    pub name: String,
    pub info: &'a TensorInfo,
    pub data: &'a [u8],
}

/// Write a complete GGUF file.
pub fn write_gguf(
    path: &Path,
    version: u32,
    alignment: u64,
    metadata: &[(String, MetadataValue)],
    tensors: &[OutputTensor<'_>],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w = GgufWriter::new(BufWriter::new(file), alignment);

    // Header
    w.write_u32(GGUF_MAGIC)?;
    w.write_u32(version)?;
    w.write_u64(tensors.len() as u64)?;
    w.write_u64(metadata.len() as u64)?;

    // Metadata
    for (key, val) in metadata {
        w.write_metadata_kv(key, val)?;
    }

    // We need to compute tensor data offsets before writing tensor info.
    // First pass: compute sizes and offsets (relative to tensor data section start).
    let mut data_offsets = Vec::with_capacity(tensors.len());
    let mut current_data_offset: u64 = 0;
    for t in tensors {
        // Each tensor's data is aligned within the data section
        current_data_offset = align_offset(current_data_offset, alignment);
        data_offsets.push(current_data_offset);
        current_data_offset += t.data.len() as u64;
    }

    // Write tensor infos
    for (i, t) in tensors.iter().enumerate() {
        w.write_tensor_info(&t.name, t.info, data_offsets[i])?;
    }

    // Pad to alignment (start of tensor data section)
    w.pad_to_alignment()?;
    let tensor_data_start = w.current_pos()?;

    // Write tensor data
    for (i, t) in tensors.iter().enumerate() {
        // Seek to correct position (tensor_data_start + offset)
        let target = tensor_data_start + data_offsets[i];
        let current = w.current_pos()?;
        if target > current {
            let padding = vec![0u8; (target - current) as usize];
            w.w.write_all(&padding)?;
        }
        w.w.write_all(t.data)?;
    }

    w.w.flush()?;
    Ok(())
}

/// Build the output GGUF from a source file, selecting layers by the given list.
/// `layer_list` contains source block numbers in the order they should appear in the output.
/// Non-block tensors are copied as-is. Block tensors are renumbered sequentially.
pub fn merge_layers(
    source: &GgufFile,
    output_path: &Path,
    layer_list: &[u32],
) -> io::Result<()> {
    let block_layers = source.block_layers();
    let max_block = source.block_count();

    // Validate layer numbers
    for &layer in layer_list {
        if layer >= max_block {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "layer {layer} does not exist in source (has layers 0..{})",
                    max_block - 1
                ),
            ));
        }
    }

    // Build metadata, updating block_count
    let new_block_count = layer_list.len() as u32;
    let mut metadata: Vec<(String, MetadataValue)> = Vec::new();
    for (key, val) in &source.metadata {
        if key.ends_with(".block_count") {
            metadata.push((key.clone(), MetadataValue::Uint32(new_block_count)));
        } else {
            metadata.push((key.clone(), val.clone()));
        }
    }

    // Collect output tensors
    let mut output_tensors: Vec<OutputTensor<'_>> = Vec::new();

    // Non-block tensors first (preserving order)
    for idx in source.non_block_tensor_indices() {
        let t = &source.tensors[idx];
        output_tensors.push(OutputTensor {
            name: t.name.clone(),
            info: t,
            data: source.tensor_data(t),
        });
    }

    // Block tensors in new order
    for (new_idx, &src_layer) in layer_list.iter().enumerate() {
        if let Some(tensor_indices) = block_layers.get(&src_layer) {
            for &ti in tensor_indices {
                let t = &source.tensors[ti];
                let suffix = t.block_suffix().unwrap();
                let new_name = format!("blk.{new_idx}.{suffix}");
                output_tensors.push(OutputTensor {
                    name: new_name,
                    info: t,
                    data: source.tensor_data(t),
                });
            }
        }
    }

    write_gguf(
        output_path,
        source.version,
        source.alignment,
        &metadata,
        &output_tensors,
    )?;

    Ok(())
}

// ── Formatting helpers ──────────────────────────────────────────────────────

pub fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
