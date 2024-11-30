use std::collections::HashMap;
use std::fmt::{self};
use std::str::FromStr;

pub use indexmap::IndexMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Nullable {
    True,
    False,
}

impl fmt::Display for Nullable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nullable::True => write!(f, "null"),
            Nullable::False => write!(f, ""),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StructField {
    pub name: Option<String>,
    pub ty: WasmType,
    pub mutable: bool,
}

impl fmt::Display for StructField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.name.clone().unwrap_or_default();
        let ty = if self.mutable {
            format!("(mut {})", self.ty)
        } else {
            format!("{}", self.ty)
        };
        writeln!(f, "(field {name} {ty}")
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Signature {
    pub params: Vec<WasmType>,
    pub result: Option<Box<WasmType>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum WasmType {
    I32,
    I64,
    F32,
    F64,
    I8,
    I31Ref,
    Anyref,
    Ref(String, Nullable),
    Array {
        mutable: bool,
        ty: Box<WasmType>,
    },
    Struct(Vec<StructField>),
    Func(Box<Signature>),
    Tag {
        name: String,
        signature: Box<Signature>,
    },
}

impl WasmType {
    pub fn is_numeric(&self) -> bool {
        use WasmType::*;
        matches!(self, I32 | I64 | F32 | I8 | I31Ref)
    }

    pub fn compatible_numeric_types(left: &Self, right: &Self) -> bool {
        use WasmType::*;
        matches!(
            (left, right),
            (I32, I32)
                | (I32, I64)
                | (I32, I8)
                | (I32, I31Ref)
                | (I64, I32)
                | (I64, I64)
                | (I64, I8)
                | (I64, I31Ref)
                | (I8, I32)
                | (I8, I64)
                | (I8, I8)
                | (I8, I31Ref)
                | (I31Ref, I32)
                | (I31Ref, I64)
                | (I31Ref, I8)
                | (I31Ref, I31Ref)
                | (F32, F32)
                | (F32, F64)
                | (F64, F32)
                | (F64, F64)
        )
    }

    pub fn broader_numeric_type(left: &Self, right: &Self) -> Self {
        use WasmType::*;
        match (left, right) {
            (I32, I32) => I32,
            (I32, I64) => I64,
            (I32, I8) => I32,
            (I32, I31Ref) => I32,
            (I64, I32) => I64,
            (I64, I64) => I64,
            (I64, I8) => I64,
            (I64, I31Ref) => I64,
            (I8, I32) => I32,
            (I8, I64) => I64,
            (I8, I8) => I8,
            (I8, I31Ref) => I31Ref,
            (I31Ref, I32) => I32,
            (I31Ref, I64) => I64,
            (I31Ref, I8) => I31Ref,
            (I31Ref, I31Ref) => I31Ref,
            (F32, F32) => F32,
            (F32, F64) => F64,
            (F64, F32) => F64,
            (F64, F64) => F64,
            _ => panic!("Both types need to be numeric. Left: {left:?}, Right: {right:?}."),
        }
    }

    pub fn convert_to_instruction(&self, to_type: &Self) -> Vec<WatInstruction> {
        use WasmType::*;
        match (self, to_type) {
            (I32, I32) => vec![],
            (I32, I64) => vec![WatInstruction::I64ExtendI32S],
            (I32, I31Ref) => vec![WatInstruction::RefI31],
            (I64, I32) => vec![WatInstruction::I32WrapI64],
            (I64, I64) => vec![],
            (I64, I31Ref) => vec![WatInstruction::I32WrapI64, WatInstruction::RefI31],
            (I31Ref, I32) => vec![WatInstruction::I31GetS],
            (I31Ref, I64) => vec![WatInstruction::I31GetS, WatInstruction::I64ExtendI32S],
            (I31Ref, I31Ref) => vec![],
            (F32, F32) => vec![],
            (F32, F64) => vec![WatInstruction::F64PromoteF32],
            (F64, F32) => vec![WatInstruction::F32DemoteF64],
            (F64, F64) => vec![],
            _ => panic!("Can't convert {self:?} to {to_type:?}"),
        }
    }

    pub fn func(params: Vec<WasmType>, result: Option<WasmType>) -> WasmType {
        WasmType::Func(Box::new(Signature {
            params,
            result: result.map(Box::new),
        }))
    }
}

impl fmt::Display for WasmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmType::I32 => write!(f, "i32"),
            WasmType::I64 => write!(f, "i64"),
            WasmType::F32 => write!(f, "f32"),
            WasmType::F64 => write!(f, "f64"),
            WasmType::I8 => write!(f, "i8"),
            WasmType::Anyref => write!(f, "anyref"),
            WasmType::I31Ref => write!(f, "i31ref"),
            WasmType::Ref(name, nullable) => write!(f, "(ref {nullable} {name})"),
            WasmType::Array { mutable, ty } => {
                let m = if *mutable {
                    format!("(mut {ty})")
                } else {
                    ty.to_string()
                };
                write!(f, "(array {m})")
            }
            WasmType::Struct(fields) => {
                write!(f, "(struct")?;
                for field in fields {
                    write!(f, "  {field}")?;
                }
                write!(f, ")")
            }
            WasmType::Func(signature) => {
                write!(f, "(func")?;
                for param in &signature.params {
                    write!(f, " (param {param})")?;
                }
                if let Some(result) = &signature.result {
                    write!(f, " (result {result})")?;
                }
                write!(f, ")")
            }
            WasmType::Tag { name, signature } => {
                write!(f, "(tag {name}")?;
                for param in &signature.params {
                    write!(f, " (param {param})")?;
                }
                if let Some(result) = &signature.result {
                    write!(f, " (result {result})")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct WasmError;

impl FromStr for WasmType {
    // TODO:implement error handling
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let str = match s {
            "i32" => Self::I32,
            "i64" => Self::I64,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "i8" => Self::I8,
            "i31ref" => Self::I31Ref,
            "anyref" => Self::Anyref,
            // this should not go here, as this crate is supposed to not
            // be tied to a certain implementation, but for now I'm leaving it here
            // TODO: fix this
            "bool" => Self::I32,
            _ => {
                // for now just handle custom types
                Self::Ref(format!("${s}"), Nullable::False)
            }
        };

        Ok(str)
    }

    type Err = WasmError;
}

#[derive(Debug, Clone)]
pub enum WatInstruction {
    Nop,
    Local {
        name: String,
        r#type: WasmType,
    },
    GlobalGet(String),
    LocalGet(String),
    LocalSet(String),
    Call(String),

    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),

    F32Neg,
    F64Neg,

    I32Eqz,
    I64Eqz,
    F32Eqz,
    F64Eqz,

    I32Eq,
    I64Eq,
    F32Eq,
    F64Eq,

    I32Ne,
    I64Ne,
    F32Ne,
    F64Ne,

    I32Add,
    I64Add,
    F32Add,
    F64Add,

    I32Sub,
    I64Sub,
    F32Sub,
    F64Sub,

    I32Mul,
    I64Mul,
    F32Mul,
    F64Mul,

    I32DivS,
    I64DivS,
    I32DivU,
    I64DivU,
    F32Div,
    F64Div,

    I32RemS,
    I64RemS,
    I32RemU,
    I64RemU,

    I32And,
    I64And,

    I32Or,
    I64Or,

    I32Xor,
    I64Xor,

    I32LtS,
    I64LtS,
    I32LtU,
    I64LtU,
    F32Lt,
    F64Lt,

    I32LeS,
    I64LeS,
    I32LeU,
    I64LeU,
    F32Le,
    F64Le,

    I32GeS,
    I64GeS,
    I32GeU,
    I64GeU,
    F32Ge,
    F64Ge,

    I32GtS,
    I64GtS,
    I32GtU,
    I64GtU,
    F32Gt,
    F64Gt,

    I32ShlS,
    I64ShlS,
    I32ShlU,
    I64ShlU,

    I32ShrS,
    I64ShrS,
    I32ShrU,
    I64ShrU,

    I64ExtendI32S,
    I32WrapI64,
    I31GetS,
    F64PromoteF32,
    F32DemoteF64,

    StructNew(String),
    StructGet(String, String),
    StructSet(String, String),
    ArrayNew(String),
    ArrayNewFixed(String, u16),
    ArrayLen,
    ArrayGet(String),
    ArrayGetU(String),
    ArraySet(String),
    RefNull(WasmType),
    Ref(String),
    RefFunc(String),
    Type(String),
    Return,
    ReturnCall(String),
    Block {
        label: String,
        instructions: Vec<WatInstruction>,
    },
    Loop {
        label: String,
        instructions: Vec<WatInstruction>,
    },
    If {
        then: Vec<WatInstruction>,
        r#else: Option<Vec<WatInstruction>>,
    },
    BrIf(String),
    Br(String),
    Empty,
    Log,
    Identifier(String),
    Drop,
    LocalTee(String),
    RefI31,
    Throw(String),
    Try {
        try_block: Box<WatInstruction>,
        catches: Vec<Box<WatInstruction>>,
        catch_all: Option<Box<WatInstruction>>,
    },
    Catch(String, Box<WatInstruction>),
    CatchAll(Box<WatInstruction>),
}

impl WatInstruction {
    pub fn local(name: impl Into<String>, r#type: WasmType) -> Box<Self> {
        Box::new(Self::Local {
            name: name.into(),
            r#type,
        })
    }

    pub fn global_get(name: impl Into<String>) -> Self {
        Self::GlobalGet(name.into())
    }

    pub fn local_get(name: impl Into<String>) -> Self {
        Self::LocalGet(name.into())
    }

    pub fn local_set(name: impl Into<String>) -> Self {
        Self::LocalSet(name.into())
    }

    pub fn local_tee(name: impl Into<String>) -> Self {
        Self::LocalTee(name.into())
    }

    pub fn call(name: impl Into<String>) -> Self {
        Self::Call(name.into())
    }

    pub fn i32_const(value: i32) -> Self {
        Self::I32Const(value)
    }

    pub fn f64_const(value: f64) -> Box<Self> {
        Box::new(Self::F64Const(value))
    }

    pub fn struct_new(name: impl Into<String>) -> Self {
        Self::StructNew(name.into())
    }

    pub fn struct_get(name: impl Into<String>, field_name: impl Into<String>) -> Self {
        Self::StructGet(name.into(), field_name.into())
    }

    pub fn struct_set(name: impl Into<String>, field_name: impl Into<String>) -> Self {
        Self::StructSet(name.into(), field_name.into())
    }

    pub fn array_new(
        name: impl Into<String>,
        _init: Box<WatInstruction>,
        _length: Box<WatInstruction>,
    ) -> Box<Self> {
        Box::new(Self::ArrayNew(name.into()))
    }

    pub fn array_get(name: impl Into<String>) -> Self {
        Self::ArrayGet(name.into())
    }

    pub fn array_get_u(name: impl Into<String>) -> Self {
        Self::ArrayGetU(name.into())
    }

    pub fn array_set(name: impl Into<String>) -> Self {
        Self::ArraySet(name.into())
    }

    pub fn ref_null(r#type: WasmType) -> Box<Self> {
        Box::new(Self::RefNull(r#type))
    }

    pub fn ref_func(name: impl Into<String>) -> Box<Self> {
        Box::new(Self::RefFunc(name.into()))
    }

    pub fn type_(name: impl Into<String>) -> Box<Self> {
        Box::new(Self::Type(name.into()))
    }

    pub fn r#return() -> Box<Self> {
        Box::new(Self::Return)
    }

    pub fn return_call(name: impl Into<String>) -> Box<Self> {
        Box::new(Self::ReturnCall(name.into()))
    }

    pub fn block(label: impl Into<String>, instructions: Vec<WatInstruction>) -> Self {
        Self::Block {
            label: label.into(),
            instructions,
        }
    }

    pub fn r#loop(label: impl Into<String>, instructions: Vec<WatInstruction>) -> Self {
        Self::Loop {
            label: label.into(),
            instructions,
        }
    }

    pub fn r#if(then: Vec<WatInstruction>, r#else: Option<Vec<WatInstruction>>) -> Self {
        Self::If { then, r#else }
    }

    pub fn br_if(label: impl Into<String>) -> Self {
        Self::BrIf(label.into())
    }

    pub fn br(label: impl Into<String>) -> Self {
        Self::Br(label.into())
    }

    pub fn empty() -> Box<Self> {
        Box::new(Self::Empty)
    }

    pub fn drop() -> Box<Self> {
        Box::new(Self::Drop)
    }

    pub fn i32_eqz() -> Box<Self> {
        Box::new(Self::I32Eqz)
    }

    pub fn ref_i31() -> Self {
        Self::RefI31
    }

    pub fn throw(label: impl Into<String>) -> Self {
        Self::Throw(label.into())
    }

    pub fn r#type(name: impl Into<String>) -> Box<Self> {
        Box::new(Self::Type(name.into()))
    }

    pub fn r#try(
        try_block: Box<Self>,
        catches: Vec<Box<Self>>,
        catch_all: Option<Box<Self>>,
    ) -> Box<Self> {
        Box::new(Self::Try {
            try_block,
            catches,
            catch_all,
        })
    }

    pub fn catch(label: impl Into<String>, instr: Box<Self>) -> Box<Self> {
        Box::new(Self::Catch(label.into(), instr))
    }

    pub fn is_return(&self) -> bool {
        matches!(self, Self::Return)
    }

    pub fn is_call(&self) -> bool {
        matches!(self, Self::Call { .. })
    }
}

impl fmt::Display for WatInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WatInstruction::F32Neg => write!(f, "(f32.neg)"),
            WatInstruction::F64Neg => write!(f, "(f64.neg)"),

            WatInstruction::I32Add => write!(f, "(i32.add)"),
            WatInstruction::I64Add => write!(f, "(i64.add)"),
            WatInstruction::F32Add => write!(f, "(f32.add)"),
            WatInstruction::F64Add => write!(f, "(f64.add)"),

            WatInstruction::I32Sub => write!(f, "(i32.sub)"),
            WatInstruction::I64Sub => write!(f, "(i64.sub)"),
            WatInstruction::F32Sub => write!(f, "(f32.sub)"),
            WatInstruction::F64Sub => write!(f, "(f64.sub)"),

            WatInstruction::I32Mul => write!(f, "(i32.mul)"),
            WatInstruction::I64Mul => write!(f, "(i64.mul)"),
            WatInstruction::F32Mul => write!(f, "(f32.mul)"),
            WatInstruction::F64Mul => write!(f, "(f64.mul)"),

            WatInstruction::I32DivS => write!(f, "(i32.div_s)"),
            WatInstruction::I64DivS => write!(f, "(i64.div_s)"),
            WatInstruction::I32DivU => write!(f, "(i32.div_u)"),
            WatInstruction::I64DivU => write!(f, "(i64.div_u)"),
            WatInstruction::F32Div => write!(f, "(f32.div)"),
            WatInstruction::F64Div => write!(f, "(f64.div)"),

            WatInstruction::I32LeS => write!(f, "(i32.le_s)"),
            WatInstruction::I64LeS => write!(f, "(i64.le_s)"),
            WatInstruction::I32LeU => write!(f, "(i32.le_u)"),
            WatInstruction::I64LeU => write!(f, "(i64.le_u)"),
            WatInstruction::F32Le => write!(f, "(f32.le)"),
            WatInstruction::F64Le => write!(f, "(f64.le)"),

            WatInstruction::I32LtS => write!(f, "(i32.lt_s)"),
            WatInstruction::I64LtS => write!(f, "(i64.lt_s)"),
            WatInstruction::I32LtU => write!(f, "(i32.lt_u)"),
            WatInstruction::I64LtU => write!(f, "(i64.lt_u)"),
            WatInstruction::F32Lt => write!(f, "(f32.lt)"),
            WatInstruction::F64Lt => write!(f, "(f64.lt)"),

            WatInstruction::I32GeS => write!(f, "(i32.ge_s)"),
            WatInstruction::I64GeS => write!(f, "(i64.ge_s)"),
            WatInstruction::I32GeU => write!(f, "(i32.ge_u)"),
            WatInstruction::I64GeU => write!(f, "(i64.ge_u)"),
            WatInstruction::F32Ge => write!(f, "(f32.ge)"),
            WatInstruction::F64Ge => write!(f, "(f64.ge)"),

            WatInstruction::I32GtS => write!(f, "(i32.gt_s)"),
            WatInstruction::I64GtS => write!(f, "(i64.gt_s)"),
            WatInstruction::I32GtU => write!(f, "(i32.gt_u)"),
            WatInstruction::I64GtU => write!(f, "(i64.gt_u)"),
            WatInstruction::F32Gt => write!(f, "(f32.gt)"),
            WatInstruction::F64Gt => write!(f, "(f64.gt)"),

            WatInstruction::I32RemS => write!(f, "(i32.rem_s)"),
            WatInstruction::I64RemS => write!(f, "(i64.rem_s)"),
            WatInstruction::I32RemU => write!(f, "(i32.rem_u)"),
            WatInstruction::I64RemU => write!(f, "(i64.rem_u)"),

            WatInstruction::I32ShlS => write!(f, "(i32.shl_s)"),
            WatInstruction::I64ShlS => write!(f, "(i64.shl_s)"),
            WatInstruction::I32ShlU => write!(f, "(i32.shl_u)"),
            WatInstruction::I64ShlU => write!(f, "(i64.shl_u)"),

            WatInstruction::I32ShrS => write!(f, "(i32.shr_s)"),
            WatInstruction::I64ShrS => write!(f, "(i64.shr_s)"),
            WatInstruction::I32ShrU => write!(f, "(i32.shr_u)"),
            WatInstruction::I64ShrU => write!(f, "(i64.shr_u)"),

            WatInstruction::I32And => write!(f, "(i32.and)"),
            WatInstruction::I64And => write!(f, "(i64.and)"),

            WatInstruction::I32Or => write!(f, "(i32.or)"),
            WatInstruction::I64Or => write!(f, "(i64.or)"),

            WatInstruction::I32Xor => write!(f, "(i32.xor)"),
            WatInstruction::I64Xor => write!(f, "(i64.xor)"),

            WatInstruction::Nop => Ok(()),
            WatInstruction::Local { name, r#type } => write!(f, "(local {} {})", name, r#type),
            WatInstruction::GlobalGet(name) => write!(f, "(global.get {})", name),
            WatInstruction::LocalGet(name) => write!(f, "(local.get {})", name),
            WatInstruction::LocalSet(name) => write!(f, "(local.set {})", name),
            WatInstruction::Call(name) => writeln!(f, "(call {})", name),

            WatInstruction::I32Const(value) => write!(f, "(i32.const {})", value),
            WatInstruction::I64Const(value) => write!(f, "(i64.const {})", value),
            WatInstruction::F32Const(value) => write!(f, "(f32.const {})", value),
            WatInstruction::F64Const(value) => write!(f, "(f64.const {})", value),

            WatInstruction::I32GeS => writeln!(f, "(i32.ge_s)"),

            WatInstruction::StructNew(name) => write!(f, "(struct.new {})", name),
            WatInstruction::StructGet(name, field) => write!(f, "(struct.get {name} {field})"),
            WatInstruction::StructSet(name, field) => write!(f, "(struct.set {name} {field})"),
            WatInstruction::ArrayNew(name) => {
                write!(f, "(array.new {name})")
            }
            WatInstruction::ArrayLen => write!(f, "(array.len)"),
            WatInstruction::ArrayGet(ty) => write!(f, "(array.get {ty})"),
            WatInstruction::ArrayGetU(ty) => write!(f, "(array.get_u {ty})"),
            WatInstruction::ArraySet(ty) => write!(f, "(array.set {ty})"),
            WatInstruction::ArrayNewFixed(typeidx, n) => {
                write!(f, "(array.new_fixed {typeidx} {n})")
            }
            WatInstruction::RefNull(r#type) => write!(f, "(ref.null {})", r#type),
            WatInstruction::RefFunc(name) => write!(f, "(ref.func ${})", name),
            WatInstruction::Return => write!(f, "return"),
            WatInstruction::ReturnCall(name) => write!(f, "(return_call {name})"),
            WatInstruction::Block {
                label,
                instructions,
            } => {
                writeln!(f, "(block {label}")?;
                for instruction in instructions {
                    writeln!(f, "  {}", instruction)?;
                }
                write!(f, ")")
            }
            WatInstruction::Loop {
                label,
                instructions,
            } => {
                writeln!(f, "(loop {label}")?;
                for instruction in instructions {
                    writeln!(f, "  {}", instruction)?;
                }
                write!(f, ")")
            }
            WatInstruction::If { then, r#else } => {
                write!(f, "(if (then")?;
                for instruction in then {
                    write!(f, " {}", instruction)?;
                }
                write!(f, ")")?;
                if let Some(else_block) = r#else {
                    write!(f, " (else")?;
                    for instruction in else_block {
                        write!(f, " {}", instruction)?;
                    }
                    write!(f, ")")?;
                }
                write!(f, ")")
            }
            WatInstruction::BrIf(label) => write!(f, "(br_if {})", label),
            WatInstruction::Br(label) => write!(f, "(br {})", label),
            WatInstruction::Type(name) => write!(f, "{}", name),
            WatInstruction::Empty => Ok(()),
            WatInstruction::Log => {
                writeln!(f, "(call $log)")
            }
            WatInstruction::Identifier(s) => write!(f, "{}", s),
            WatInstruction::Ref(s) => write!(f, "(ref ${})", s),
            WatInstruction::Drop => writeln!(f, "(drop)"),
            WatInstruction::LocalTee(name) => write!(f, "(local.tee {})", name),

            WatInstruction::I32Eqz => write!(f, "(i32.eqz)"),
            WatInstruction::I64Eqz => write!(f, "(i64.eqz)"),
            WatInstruction::F32Eqz => write!(f, "(f32.eqz)"),
            WatInstruction::F64Eqz => write!(f, "(f64.eqz)"),

            WatInstruction::RefI31 => write!(f, "(ref.i31)"),
            WatInstruction::Throw(label) => write!(f, "(throw {label})"),
            WatInstruction::Try {
                try_block,
                catches,
                catch_all,
            } => {
                writeln!(
                    f,
                    "\ntry\n{try_block}{}{}\nend",
                    catches
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<String>>()
                        .join(""),
                    catch_all
                        .clone()
                        .map(|c| c.to_string())
                        .unwrap_or("".to_string())
                )
            }
            WatInstruction::Catch(label, instr) => writeln!(f, "\ncatch {label}\n{instr}"),
            WatInstruction::CatchAll(instr) => writeln!(f, "\ncatch_all\n{instr}"),
            WatInstruction::I64ExtendI32S => writeln!(f, "(i64.extend_32_s)"),
            WatInstruction::I32WrapI64 => writeln!(f, "(i32.wrap_i64)"),
            WatInstruction::I31GetS => writeln!(f, "(i31.get_s)"),
            WatInstruction::F64PromoteF32 => writeln!(f, "(f64.promote_f32)"),
            WatInstruction::F32DemoteF64 => writeln!(f, "(f32.demote_f64)"),

            WatInstruction::I32Eq => writeln!(f, "(i32.eq)"),
            WatInstruction::I64Eq => writeln!(f, "(i64.eq)"),
            WatInstruction::F32Eq => writeln!(f, "(f32.eq)"),
            WatInstruction::F64Eq => writeln!(f, "(f64.eq)"),

            WatInstruction::I32Ne => writeln!(f, "(i32.ne)"),
            WatInstruction::I64Ne => writeln!(f, "(i64.ne)"),
            WatInstruction::F32Ne => writeln!(f, "(f32.ne)"),
            WatInstruction::F64Ne => writeln!(f, "(f64.ne)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WatFunction {
    pub name: String,
    pub params: Vec<(String, WasmType)>,
    pub return_type: Option<WasmType>,
    pub locals: HashMap<String, WasmType>,
    pub locals_counters: HashMap<String, u32>,
    pub body: Vec<Box<WatInstruction>>,
}

impl WatFunction {
    pub fn new(name: impl Into<String>) -> Self {
        WatFunction {
            name: name.into(),
            params: Vec::new(),
            return_type: None,
            locals: HashMap::new(),
            locals_counters: HashMap::new(),
            body: Vec::new(),
        }
    }

    pub fn add_param(&mut self, name: impl Into<String>, type_: &WasmType) {
        self.params.push((name.into(), type_.clone()));
    }

    pub fn set_return_type(&mut self, type_: WasmType) {
        self.return_type = Some(type_);
    }

    pub fn add_local_exact(&mut self, name: impl Into<String>, r#type: WasmType) {
        self.locals.insert(name.into(), r#type);
    }

    pub fn add_local(&mut self, name: impl Into<String>, r#type: WasmType) -> String {
        let name = name.into();

        let counter = self.locals_counters.entry(name.clone()).or_insert(0);
        *counter += 1;
        let name = format!("{name}-{counter}");
        self.locals.insert(name.clone(), r#type);

        name
    }

    pub fn add_instruction(&mut self, instruction: WatInstruction) {
        self.body.push(Box::new(instruction));
    }
}

impl fmt::Display for WatFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(func ${}", self.name)?;
        for (name, type_) in &self.params {
            write!(f, " (param {} {})", name, type_)?;
        }
        if let Some(return_type) = &self.return_type {
            write!(f, " (result {})", return_type)?;
        }
        writeln!(f)?;
        for (name, type_) in &self.locals {
            writeln!(f, "  (local {} {})", name, type_)?;
        }
        for instruction in &self.body {
            writeln!(f, "  {}", instruction)?;
        }
        writeln!(f, ")")
    }
}

#[derive(Debug, Clone, Default)]
pub struct WatModule {
    pub tags: IndexMap<String, String>,
    pub types: IndexMap<String, WasmType>,
    pub imports: Vec<(String, String, WasmType)>,
    pub functions: Vec<WatFunction>,
    // TODO: changet it to a struct
    pub exports: Vec<(String, String, String)>,
    pub globals: Vec<(String, WasmType, WatInstruction)>,
    pub data: Vec<(usize, String)>,
    pub data_offset: usize,
    pub data_offsets: HashMap<String, usize>,
    pub memories: HashMap<String, (i32, Option<i32>)>,
}

impl WatModule {
    pub fn new() -> Self {
        WatModule {
            tags: IndexMap::new(),
            types: IndexMap::new(),
            imports: Vec::new(),
            functions: Vec::new(),
            exports: Vec::new(),
            globals: Vec::new(),
            data: Vec::new(),
            data_offset: 100,
            data_offsets: HashMap::new(),
            memories: HashMap::new(),
        }
    }

    pub fn add_function(&mut self, function: WatFunction) {
        self.functions.push(function);
    }

    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut WatFunction> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    pub fn add_type(&mut self, name: impl Into<String>, ty: WasmType) {
        self.types.insert(name.into(), ty);
    }

    pub fn add_data(&mut self, content: String) -> (usize, usize) {
        let len = content.len();
        let offset = self.data_offset;
        if let Some(offset) = self.data_offsets.get(&content) {
            (*offset, len)
        } else {
            self.data.push((offset, content.clone()));
            self.data_offsets.insert(content, offset);
            self.data_offset += if len % 4 == 0 {
                len
            } else {
                // some runtimes expect all data aligned to 4 bytes
                len + (4 - len % 4)
            };

            (offset, len)
        }
    }

    pub fn add_memory(&mut self, label: impl Into<String>, size: i32, max_size: Option<i32>) {
        self.memories.insert(label.into(), (size, max_size));
    }

    pub fn add_export(
        &mut self,
        export_name: impl Into<String>,
        export_type: impl Into<String>,
        name: impl Into<String>,
    ) {
        self.exports
            .push((export_name.into(), export_type.into(), name.into()));
    }

    pub fn add_import(
        &mut self,
        namespace: impl Into<String>,
        name: impl Into<String>,
        ty: WasmType,
    ) {
        self.imports.push((namespace.into(), name.into(), ty));
    }
}

impl fmt::Display for WatModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(module")?;
        // Types
        for (name, ty) in &self.types {
            writeln!(f, "  (type {name} {ty})")?;
        }

        // Tags
        for (label, typeidx) in &self.tags {
            writeln!(f, "  (tag {label} (type {typeidx}))")?;
        }

        // Imports
        for (module, name, type_) in &self.imports {
            writeln!(f, "  (import \"{}\" \"{}\" {})", module, name, type_)?;
        }

        // Memories
        for (label, (size, max_size)) in &self.memories {
            let max_size = if let Some(max_size) = max_size {
                format!("{max_size}")
            } else {
                "".into()
            };
            writeln!(f, "  (memory {label} {size} {max_size})")?;
        }

        // Data
        for (offset, data) in &self.data {
            write!(f, "  (data (i32.const {}) \"", offset)?;
            // TODO: this escaping should be done when inserting the data
            for &byte in data.as_bytes() {
                match byte {
                    b'"' => write!(f, "\\\"")?,
                    b'\\' => write!(f, "\\\\")?,
                    b'\n' => write!(f, "\\n")?,
                    b'\r' => write!(f, "\\r")?,
                    b'\t' => write!(f, "\\t")?,
                    _ if byte.is_ascii_graphic() || byte == b' ' => write!(f, "{}", byte as char)?,
                    _ => write!(f, "\\{:02x}", byte)?,
                }
            }
            writeln!(f, "\")")?;
        }

        // Globals
        for (name, type_, init) in &self.globals {
            writeln!(f, "  (global ${} {} {})", name, type_, init)?;
        }

        // Function declarations
        for function in &self.functions {
            writeln!(f, "(elem declare func ${})", function.name)?;
        }

        // Functions
        for function in &self.functions {
            write!(f, "  {}", function)?;
        }

        // Exports
        for (name, export_type, internal_name) in &self.exports {
            writeln!(f, "  (export \"{name}\" ({export_type} {internal_name}))",)?;
        }

        writeln!(f, ")")?;

        Ok(())
    }
}
