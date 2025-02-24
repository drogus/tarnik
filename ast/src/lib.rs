use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{self, Formatter};
use std::rc::Rc;
use std::str::FromStr;

pub use indexmap::IndexMap;
use slotmap::SlotMap;

slotmap::new_key_type! {
    pub struct FunctionKey;
}

pub type InstructionsList = Vec<WatInstruction>;
pub type InstructionsListWrapped = Rc<RefCell<InstructionsList>>;

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
        let mut name = self.name.clone().unwrap_or_default();
        let ty = if self.mutable {
            format!("(mut {})", self.ty)
        } else {
            format!("{}", self.ty)
        };
        // TODO: it would be nice to actually fix it downstream
        if !name.starts_with("$") {
            name = format!("${name}");
        }
        write!(f, "(field {name} {ty})")
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct Signature {
    pub params: Vec<(Option<String>, WasmType)>,
    pub result: Option<WasmType>,
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
    NullRef,
    Ref(String, Nullable),
    Array {
        mutable: bool,
        ty: Box<WasmType>,
    },
    Struct(Vec<StructField>),
    Func {
        name: Option<String>,
        signature: Box<Signature>,
    },
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

    pub fn r#ref(name: impl Into<String>) -> Self {
        Self::Ref(name.into(), Nullable::False)
    }

    pub fn ref_null(name: impl Into<String>) -> Self {
        Self::Ref(name.into(), Nullable::True)
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

    pub fn convert_to_instruction(&self, to_type: &Self) -> InstructionsList {
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

    pub fn func(
        name: Option<String>,
        params: Vec<(Option<String>, WasmType)>,
        result: Option<WasmType>,
    ) -> WasmType {
        WasmType::Func {
            name,
            signature: Box::new(Signature { params, result }),
        }
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
            WasmType::NullRef => write!(f, "nullref"),
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
                writeln!(f, "(struct")?;
                for field in fields {
                    writeln!(f, "  {field}")?;
                }
                writeln!(f, ")")
            }
            WasmType::Func { name, signature } => {
                let name = name.clone().unwrap_or_default();
                write!(f, "(func {name}")?;
                for (name, ty) in &signature.params {
                    let param_name = name.clone().unwrap_or_default();
                    write!(f, " (param {param_name} {ty})")?;
                }
                if let Some(result) = &signature.result {
                    write!(f, " (result {result})")?;
                }
                writeln!(f, ")")
            }
            WasmType::Tag { name, signature } => {
                write!(f, "(tag {name}")?;
                for (name, ty) in &signature.params {
                    let param_name = name.clone().unwrap_or_default();
                    write!(f, " (param {param_name} {ty})")?;
                }
                if let Some(result) = &signature.result {
                    write!(f, " (result {result})")?;
                }
                writeln!(f, ")")
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
            "nullref" => Self::NullRef,
            "null" => Self::NullRef,
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

#[derive(Debug, Clone, PartialEq)]
pub enum WatInstruction {
    Nop,
    Local {
        name: String,
        ty: WasmType,
    },
    GlobalGet(String),
    GlobalSet(String),
    LocalGet(String),
    LocalSet(String),
    Call(String),
    CallRef(String),

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

    I32Shl,
    I64Shl,

    I32ShrS,
    I64ShrS,
    I32ShrU,
    I64ShrU,

    I64ExtendI32S,
    I32WrapI64,
    F64PromoteF32,
    F32DemoteF64,

    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,

    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,

    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,

    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,

    I32ReinterpretF32,
    F32ReinterpretI32,
    I64ReinterpretF64,
    F64ReinterpretI64,

    I31GetS,
    I31GetU,

    I32Store(Option<String>),
    I64Store(Option<String>),
    F32Store(Option<String>),
    F64Store(Option<String>),
    I32Store8(Option<String>),
    I32Store16(Option<String>),
    I64Store8(Option<String>),
    I64Store16(Option<String>),
    I64Store32(Option<String>),

    I32Load(Option<String>),
    I64Load(Option<String>),
    F32Load(Option<String>),
    F64Load(Option<String>),
    I32Load8S(Option<String>),
    I32Load8U(Option<String>),
    I32Load16S(Option<String>),
    I32Load16U(Option<String>),
    I64Load8S(Option<String>),
    I64Load8U(Option<String>),
    I64Load16S(Option<String>),
    I64Load16U(Option<String>),
    I64Load32S(Option<String>),
    I64Load32U(Option<String>),

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
    RefCast(WasmType),
    RefTest(WasmType),
    Ref(String),
    RefFunc(String),
    Type(String),
    Return,
    ReturnCall(String),
    Block {
        label: String,
        signature: Signature,
        instructions: InstructionsListWrapped,
    },
    Loop {
        label: String,
        instructions: InstructionsListWrapped,
    },
    If {
        then: InstructionsListWrapped,
        r#else: Option<InstructionsListWrapped>,
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
        try_block: InstructionsListWrapped,
        catches: Vec<(String, InstructionsListWrapped)>,
        catch_all: Option<InstructionsListWrapped>,
    },
    RefEq,
}

impl WatInstruction {
    pub fn local(name: impl Into<String>, ty: WasmType) -> Self {
        Self::Local {
            name: name.into(),
            ty,
        }
    }

    pub fn global_get(name: impl Into<String>) -> Self {
        Self::GlobalGet(name.into())
    }

    pub fn global_set(name: impl Into<String>) -> Self {
        Self::GlobalSet(name.into())
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

    pub fn call_ref(name: impl Into<String>) -> Self {
        Self::CallRef(name.into())
    }

    pub fn i32_const(value: i32) -> Self {
        Self::I32Const(value)
    }

    pub fn f64_const(value: f64) -> Self {
        Self::F64Const(value)
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

    pub fn array_new(name: impl Into<String>) -> Self {
        Self::ArrayNew(name.into())
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

    pub fn ref_null(r#type: WasmType) -> Self {
        Self::RefNull(r#type)
    }

    pub fn ref_null_any() -> Self {
        Self::RefNull(WasmType::Anyref)
    }

    pub fn ref_func(name: impl Into<String>) -> Self {
        Self::RefFunc(name.into())
    }

    pub fn ref_cast(t: WasmType) -> Self {
        Self::RefCast(t)
    }

    pub fn ref_test(t: WasmType) -> Self {
        Self::RefTest(t)
    }

    pub fn type_(name: impl Into<String>) -> Self {
        Self::Type(name.into())
    }

    pub fn r#return() -> Self {
        Self::Return
    }

    pub fn return_call(name: impl Into<String>) -> Self {
        Self::ReturnCall(name.into())
    }

    pub fn block(
        label: impl Into<String>,
        signature: Signature,
        instructions: InstructionsList,
    ) -> Self {
        Self::Block {
            label: label.into(),
            signature,
            instructions: Rc::new(RefCell::new(instructions)),
        }
    }

    pub fn r#loop(label: impl Into<String>, instructions: InstructionsList) -> Self {
        Self::Loop {
            label: label.into(),
            instructions: Rc::new(RefCell::new(instructions)),
        }
    }

    pub fn r#if(then: InstructionsList, r#else: Option<InstructionsList>) -> Self {
        Self::If {
            then: Rc::new(RefCell::new(then)),
            r#else: r#else.map(|e| Rc::new(RefCell::new(e))),
        }
    }

    pub fn br_if(label: impl Into<String>) -> Self {
        Self::BrIf(label.into())
    }

    pub fn br(label: impl Into<String>) -> Self {
        Self::Br(label.into())
    }

    pub fn empty() -> Self {
        Self::Empty
    }

    pub fn drop() -> Self {
        Self::Drop
    }

    pub fn i32_eqz() -> Self {
        Self::I32Eqz
    }

    pub fn ref_i31() -> Self {
        Self::RefI31
    }

    pub fn throw(label: impl Into<String>) -> Self {
        Self::Throw(label.into())
    }

    pub fn r#type(name: impl Into<String>) -> Self {
        Self::Type(name.into())
    }

    pub fn r#try(
        try_block: InstructionsList,
        catches: Vec<(String, InstructionsList)>,
        catch_all: Option<InstructionsList>,
    ) -> Self {
        Self::Try {
            try_block: Rc::new(RefCell::new(try_block)),
            catches: catches
                .into_iter()
                .map(|(label, instructions)| (label, Rc::new(RefCell::new(instructions))))
                .collect(),
            catch_all: catch_all.map(|c| Rc::new(RefCell::new(c))),
        }
    }

    pub fn is_return(&self) -> bool {
        matches!(self, Self::Return)
    }

    pub fn is_call(&self) -> bool {
        matches!(self, Self::Call { .. })
    }

    pub fn call_name(&self) -> Option<&'_ str> {
        if self.is_call() {
            if let Self::Call(name) = self {
                return Some(name);
            }
        }
        None
    }

    pub fn is_block_type(&self) -> bool {
        self.is_block()
            || self.is_loop()
            || matches!(self, Self::Try { .. })
            || matches!(self, Self::If { .. })
    }

    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block { .. })
    }

    pub fn is_loop(&self) -> bool {
        matches!(self, Self::Loop { .. })
    }

    pub fn block_label(&self) -> Option<String> {
        match self {
            WatInstruction::Block { label, .. } => Some(label.clone()),
            WatInstruction::Loop { label, .. } => Some(label.clone()),
            _ => None,
        }
    }

    pub fn loop_label(&self) -> Option<String> {
        match self {
            WatInstruction::Block { label, .. } => Some(label.clone()),
            WatInstruction::Loop { label, .. } => Some(label.clone()),
            _ => None,
        }
    }
}

impl fmt::Display for WatInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WatInstruction::F32Neg => writeln!(f, "f32.neg"),
            WatInstruction::F64Neg => writeln!(f, "f64.neg"),

            WatInstruction::I32Add => writeln!(f, "i32.add"),
            WatInstruction::I64Add => writeln!(f, "i64.add"),
            WatInstruction::F32Add => writeln!(f, "f32.add"),
            WatInstruction::F64Add => writeln!(f, "f64.add"),

            WatInstruction::I32Sub => writeln!(f, "i32.sub"),
            WatInstruction::I64Sub => writeln!(f, "i64.sub"),
            WatInstruction::F32Sub => writeln!(f, "f32.sub"),
            WatInstruction::F64Sub => writeln!(f, "f64.sub"),

            WatInstruction::I32Mul => writeln!(f, "i32.mul"),
            WatInstruction::I64Mul => writeln!(f, "i64.mul"),
            WatInstruction::F32Mul => writeln!(f, "f32.mul"),
            WatInstruction::F64Mul => writeln!(f, "f64.mul"),

            WatInstruction::I32DivS => writeln!(f, "i32.div_s"),
            WatInstruction::I64DivS => writeln!(f, "i64.div_s"),
            WatInstruction::I32DivU => writeln!(f, "i32.div_u"),
            WatInstruction::I64DivU => writeln!(f, "i64.div_u"),
            WatInstruction::F32Div => writeln!(f, "f32.div"),
            WatInstruction::F64Div => writeln!(f, "f64.div"),

            WatInstruction::I32LeS => writeln!(f, "i32.le_s"),
            WatInstruction::I64LeS => writeln!(f, "i64.le_s"),
            WatInstruction::I32LeU => writeln!(f, "i32.le_u"),
            WatInstruction::I64LeU => writeln!(f, "i64.le_u"),
            WatInstruction::F32Le => writeln!(f, "f32.le"),
            WatInstruction::F64Le => writeln!(f, "f64.le"),

            WatInstruction::I32LtS => writeln!(f, "i32.lt_s"),
            WatInstruction::I64LtS => writeln!(f, "i64.lt_s"),
            WatInstruction::I32LtU => writeln!(f, "i32.lt_u"),
            WatInstruction::I64LtU => writeln!(f, "i64.lt_u"),
            WatInstruction::F32Lt => writeln!(f, "f32.lt"),
            WatInstruction::F64Lt => writeln!(f, "f64.lt"),

            WatInstruction::I32GeS => writeln!(f, "i32.ge_s"),
            WatInstruction::I64GeS => writeln!(f, "i64.ge_s"),
            WatInstruction::I32GeU => writeln!(f, "i32.ge_u"),
            WatInstruction::I64GeU => writeln!(f, "i64.ge_u"),
            WatInstruction::F32Ge => writeln!(f, "f32.ge"),
            WatInstruction::F64Ge => writeln!(f, "f64.ge"),

            WatInstruction::I32GtS => writeln!(f, "i32.gt_s"),
            WatInstruction::I64GtS => writeln!(f, "i64.gt_s"),
            WatInstruction::I32GtU => writeln!(f, "i32.gt_u"),
            WatInstruction::I64GtU => writeln!(f, "i64.gt_u"),
            WatInstruction::F32Gt => writeln!(f, "f32.gt"),
            WatInstruction::F64Gt => writeln!(f, "f64.gt"),

            WatInstruction::I32RemS => writeln!(f, "i32.rem_s"),
            WatInstruction::I64RemS => writeln!(f, "i64.rem_s"),
            WatInstruction::I32RemU => writeln!(f, "i32.rem_u"),
            WatInstruction::I64RemU => writeln!(f, "i64.rem_u"),

            WatInstruction::I32Shl => writeln!(f, "i32.shl"),
            WatInstruction::I64Shl => writeln!(f, "i64.shl"),

            WatInstruction::I32ShrS => writeln!(f, "i32.shr_s"),
            WatInstruction::I64ShrS => writeln!(f, "i64.shr_s"),
            WatInstruction::I32ShrU => writeln!(f, "i32.shr_u"),
            WatInstruction::I64ShrU => writeln!(f, "i64.shr_u"),

            WatInstruction::I32And => writeln!(f, "i32.and"),
            WatInstruction::I64And => writeln!(f, "i64.and"),

            WatInstruction::I32Or => writeln!(f, "i32.or"),
            WatInstruction::I64Or => writeln!(f, "i64.or"),

            WatInstruction::I32Xor => writeln!(f, "i32.xor"),
            WatInstruction::I64Xor => writeln!(f, "i64.xor"),

            WatInstruction::I32Store(label) => memory_op(f, "i32.store", label),
            WatInstruction::I64Store(label) => memory_op(f, "i64.store", label),
            WatInstruction::F32Store(label) => memory_op(f, "f32.store", label),
            WatInstruction::F64Store(label) => memory_op(f, "f64.store", label),
            WatInstruction::I32Store8(label) => memory_op(f, "i32.store8", label),
            WatInstruction::I32Store16(label) => memory_op(f, "i32.store16", label),
            WatInstruction::I64Store8(label) => memory_op(f, "i64.store8", label),
            WatInstruction::I64Store16(label) => memory_op(f, "i64.store16", label),
            WatInstruction::I64Store32(label) => memory_op(f, "i64.store32", label),

            WatInstruction::I32Load(label) => memory_op(f, "i32.load", label),
            WatInstruction::I64Load(label) => memory_op(f, "i64.load", label),
            WatInstruction::F32Load(label) => memory_op(f, "f32.load", label),
            WatInstruction::F64Load(label) => memory_op(f, "f64.load", label),
            WatInstruction::I32Load8S(label) => memory_op(f, "i32.load8_s", label),
            WatInstruction::I32Load8U(label) => memory_op(f, "i32.load8_u", label),
            WatInstruction::I32Load16S(label) => memory_op(f, "i32.load16_s", label),
            WatInstruction::I32Load16U(label) => memory_op(f, "i32.load16_u", label),
            WatInstruction::I64Load8S(label) => memory_op(f, "i64.load8_s", label),
            WatInstruction::I64Load8U(label) => memory_op(f, "i64.load8_u", label),
            WatInstruction::I64Load16S(label) => memory_op(f, "i64.load16_s", label),
            WatInstruction::I64Load16U(label) => memory_op(f, "i64.load16_u", label),
            WatInstruction::I64Load32S(label) => memory_op(f, "i64.load32_s", label),
            WatInstruction::I64Load32U(label) => memory_op(f, "i64.load32_u", label),

            WatInstruction::Nop => writeln!(f, "nop"),
            WatInstruction::Local { name, ty } => writeln!(f, "local {} {}", name, ty),
            WatInstruction::GlobalGet(name) => writeln!(f, "global.get {}", name),
            WatInstruction::GlobalSet(name) => writeln!(f, "global.set {}", name),
            WatInstruction::LocalGet(name) => writeln!(f, "local.get {}", name),
            WatInstruction::LocalSet(name) => writeln!(f, "local.set {}", name),
            WatInstruction::Call(name) => writeln!(f, "call {}", name),
            WatInstruction::CallRef(name) => writeln!(f, "call_ref {}", name),

            WatInstruction::I32Const(value) => writeln!(f, "i32.const {}", value),
            WatInstruction::I64Const(value) => writeln!(f, "i64.const {}", value),
            WatInstruction::F32Const(value) => writeln!(f, "f32.const {}", value),
            WatInstruction::F64Const(value) => writeln!(f, "f64.const {}", value),

            WatInstruction::StructNew(name) => writeln!(f, "struct.new {}", name),
            WatInstruction::StructGet(name, field) => writeln!(f, "struct.get {name} {field}"),
            WatInstruction::StructSet(name, field) => writeln!(f, "struct.set {name} {field}"),
            WatInstruction::ArrayNew(name) => writeln!(f, "array.new {name}"),
            WatInstruction::ArrayLen => writeln!(f, "array.len"),
            WatInstruction::ArrayGet(ty) => writeln!(f, "array.get {ty}"),
            WatInstruction::ArrayGetU(ty) => writeln!(f, "array.get_u {ty}"),
            WatInstruction::ArraySet(ty) => writeln!(f, "array.set {ty}"),
            WatInstruction::ArrayNewFixed(typeidx, n) => {
                writeln!(f, "array.new_fixed {typeidx} {n}")
            }
            WatInstruction::RefNull(ty) => {
                let ty_str = match ty {
                    WasmType::I31Ref => "i31",
                    WasmType::Anyref => "any",
                    WasmType::Ref(name, _) => name,
                    t => panic!("Can't generate ref.null from {t:#?}"),
                };

                writeln!(f, "ref.null {ty_str}")
            }
            WatInstruction::RefFunc(name) => writeln!(f, "ref.func {}", name),
            WatInstruction::Return => writeln!(f, "return"),
            WatInstruction::ReturnCall(name) => writeln!(f, "return_call {name}"),
            WatInstruction::Block {
                label,
                signature: _,
                instructions,
            } => {
                writeln!(f, "block {label}")?;
                for instruction in instructions.borrow().iter() {
                    writeln!(f, "{}", indent_instruction(instruction))?;
                }
                writeln!(f, "end")
            }
            WatInstruction::Loop {
                label,
                instructions,
            } => {
                writeln!(f, "loop {label}")?;
                for instruction in instructions.borrow().iter() {
                    writeln!(f, "{}", indent_instruction(instruction))?;
                }
                writeln!(f, "end")
            }
            WatInstruction::If { then, r#else } => {
                writeln!(f, "if")?;
                for instruction in then.borrow().iter() {
                    write!(f, "{}", indent_instruction(instruction))?;
                }
                if let Some(else_block) = r#else {
                    writeln!(f, "else")?;
                    for instruction in else_block.borrow().iter() {
                        write!(f, "{}", indent_instruction(instruction))?;
                    }
                }
                writeln!(f, "end")
            }
            WatInstruction::BrIf(label) => writeln!(f, "br_if {}", label),
            WatInstruction::Br(label) => writeln!(f, "br {}", label),
            WatInstruction::Type(name) => write!(f, "{}", name),
            WatInstruction::Empty => Ok(()),
            WatInstruction::Log => {
                writeln!(f, "call $log")
            }
            WatInstruction::Identifier(s) => write!(f, "{}", s),
            WatInstruction::Ref(s) => write!(f, "ref ${}", s),
            WatInstruction::Drop => writeln!(f, "drop"),
            WatInstruction::LocalTee(name) => writeln!(f, "local.tee {}", name),

            WatInstruction::I32Eqz => writeln!(f, "i32.eqz"),
            WatInstruction::I64Eqz => writeln!(f, "i64.eqz"),
            WatInstruction::F32Eqz => writeln!(f, "f32.eqz"),
            WatInstruction::F64Eqz => writeln!(f, "f64.eqz"),

            WatInstruction::RefI31 => writeln!(f, "ref.i31"),
            WatInstruction::Throw(label) => writeln!(f, "throw {label}"),
            WatInstruction::Try {
                try_block,
                catches,
                catch_all,
            } => {
                let try_block_str = try_block
                    .borrow()
                    .iter()
                    .map(|i| indent_instruction(i))
                    .collect::<Vec<String>>()
                    .join("");

                let catches_str = catches
                    .iter()
                    .map(|(name, c)| {
                        format!(
                            "catch {name}\n{}",
                            c.borrow()
                                .iter()
                                .map(|i| indent_instruction(i))
                                .collect::<Vec<String>>()
                                .join("")
                        )
                    })
                    .collect::<Vec<String>>()
                    .join("");

                let catch_all_str = catch_all
                    .clone()
                    .map(|c| {
                        format!(
                            "catch_all\n{}",
                            c.borrow()
                                .iter()
                                .map(|i| indent_instruction(i))
                                .collect::<Vec<String>>()
                                .join("")
                        )
                    })
                    .unwrap_or("".to_string());

                writeln!(f, "try\n{try_block_str}{catches_str}{catch_all_str}end")
            }
            WatInstruction::I64ExtendI32S => writeln!(f, "i64.extend_i32_s"),
            WatInstruction::I32WrapI64 => writeln!(f, "i32.wrap_i64"),
            WatInstruction::I31GetS => writeln!(f, "i31.get_s"),
            WatInstruction::F64PromoteF32 => writeln!(f, "f64.promote_f32"),
            WatInstruction::F32DemoteF64 => writeln!(f, "f32.demote_f64"),

            WatInstruction::I32Eq => writeln!(f, "i32.eq"),
            WatInstruction::I64Eq => writeln!(f, "i64.eq"),
            WatInstruction::F32Eq => writeln!(f, "f32.eq"),
            WatInstruction::F64Eq => writeln!(f, "f64.eq"),

            WatInstruction::I32Ne => writeln!(f, "i32.ne"),
            WatInstruction::I64Ne => writeln!(f, "i64.ne"),
            WatInstruction::F32Ne => writeln!(f, "f32.ne"),
            WatInstruction::F64Ne => writeln!(f, "f64.ne"),

            WatInstruction::F32ConvertI32S => writeln!(f, "f32.convert_i32_s"),
            WatInstruction::F32ConvertI32U => writeln!(f, "f32.convert_i32_u"),
            WatInstruction::F32ConvertI64S => writeln!(f, "f32.convert_i64_s"),
            WatInstruction::F32ConvertI64U => writeln!(f, "f32.convert_i64_u"),

            WatInstruction::F64ConvertI32S => writeln!(f, "f64.convert_i32_s"),
            WatInstruction::F64ConvertI32U => writeln!(f, "f64.convert_i32_u"),
            WatInstruction::F64ConvertI64S => writeln!(f, "f64.convert_i64_s"),
            WatInstruction::F64ConvertI64U => writeln!(f, "f64.convert_i64_u"),

            WatInstruction::I32TruncF32S => writeln!(f, "i32.trunc_f32_s"),
            WatInstruction::I32TruncF32U => writeln!(f, "i32.trunc_f32_u"),
            WatInstruction::I32TruncF64S => writeln!(f, "i32.trunc_f64_s"),
            WatInstruction::I32TruncF64U => writeln!(f, "i32.trunc_f64_u"),

            WatInstruction::I64TruncF32S => writeln!(f, "i64.trunc_f32_s"),
            WatInstruction::I64TruncF32U => writeln!(f, "i64.trunc_f32_u"),
            WatInstruction::I64TruncF64S => writeln!(f, "i64.trunc_f64_s"),
            WatInstruction::I64TruncF64U => writeln!(f, "i64.trunc_f64_u"),

            WatInstruction::F32ReinterpretI32 => writeln!(f, "f32.reinterpret_i32"),
            WatInstruction::I32ReinterpretF32 => writeln!(f, "i32.reinterpret_f32"),
            WatInstruction::F64ReinterpretI64 => writeln!(f, "f64.reinterpret_i64"),
            WatInstruction::I64ReinterpretF64 => writeln!(f, "i64.reinterpret_f64"),

            WatInstruction::I31GetU => writeln!(f, "i31.get_u"),
            WatInstruction::RefCast(ty) => writeln!(f, "ref.cast {ty}"),
            WatInstruction::RefTest(ty) => {
                let ty_str = match ty {
                    WasmType::I31Ref => "i31ref",
                    WasmType::Anyref => "anyref",
                    WasmType::NullRef => "nullref",
                    t => &t.to_string(),
                };
                writeln!(f, "ref.test {ty_str}")
            }
            WatInstruction::RefEq => writeln!(f, "ref.eq"),
        }
    }
}

fn indent_instruction(instruction: &WatInstruction) -> String {
    indent_str(instruction.to_string())
}

fn indent_str(s: String) -> String {
    let has_newline = s.ends_with('\n');
    let result = s
        .lines()
        .map(|l| format!("  {l}"))
        .collect::<Vec<String>>()
        .join("\n");

    if has_newline {
        format!("{result}\n")
    } else {
        result
    }
}

#[derive(Debug, Clone)]
pub struct WatFunction {
    pub name: String,
    pub params: Vec<(Option<String>, WasmType)>,
    pub results: Vec<WasmType>,
    pub locals: IndexMap<String, WasmType>,
    pub locals_counters: HashMap<String, u32>,
    pub body: InstructionsListWrapped,
}

impl PartialEq for WatFunction {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.params == other.params
            && self.results == other.results
            && self.locals == other.locals
            && self.body == other.body
    }
}

impl WatFunction {
    pub fn new(name: impl Into<String>) -> Self {
        WatFunction {
            name: name.into(),
            params: Vec::new(),
            results: vec![],
            locals: IndexMap::new(),
            locals_counters: HashMap::new(),
            body: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn set_body(&mut self, body: InstructionsList) {
        self.body = Rc::new(RefCell::new(body));
    }

    pub fn replace(&mut self, other: WatFunction) {
        self.name = other.name;
        self.params = other.params;
        self.results = other.results;
        self.locals = other.locals;
        self.locals_counters = other.locals_counters;
        self.body = other.body;
    }

    pub fn add_param(&mut self, name: impl Into<String>, type_: &WasmType) {
        self.params.push((Some(name.into()), type_.clone()));
    }

    pub fn set_results(&mut self, results: Vec<WasmType>) {
        self.results = results;
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
        self.body.borrow_mut().push(instruction);
    }

    pub fn add_instructions(&mut self, mut instructions: InstructionsList) {
        self.body.borrow_mut().append(&mut instructions);
    }

    pub fn prepend_instructions(&mut self, instructions: InstructionsList) {
        for instruction in instructions.into_iter().rev() {
            self.body.borrow_mut().insert(0, instruction);
        }
    }

    pub fn has_results(&self) -> bool {
        !self.results.is_empty()
    }

    pub fn add_result(&mut self, ty: WasmType) {
        self.results.push(ty);
    }

    pub fn local_exists(&self, search_for: &str) -> bool {
        self.locals.iter().any(|(name, _)| *name == search_for)
    }
}

pub mod cursor;
pub mod wat_converter;
pub mod test_helpers;

#[cfg(test)]
use test_helpers::*;

impl fmt::Display for WatFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  (func ${}", self.name)?;
        for (name, type_) in &self.params {
            let param_str = vec![name.clone(), Some(type_.to_string())]
                .into_iter()
                .flatten()
                .collect::<Vec<String>>()
                .join(" ");
            writeln!(f, "    (param {param_str})")?;
        }
        if self.has_results() {
            write!(f, "    (result ")?;
            for result in &self.results {
                write!(f, "{} ", result)?;
            }
            writeln!(f, ")")?;
        }
        for (name, type_) in &self.locals {
            writeln!(f, "    (local {} {})", name, type_)?;
        }
        writeln!(f)?;
        for instruction in self.body.borrow().iter() {
            write!(f, "{}", indent_str(indent_instruction(instruction)))?;
        }
        writeln!(f, "  )\n")
    }
}

#[derive(Debug, Clone)]
pub struct Global {
    pub name: String,
    pub ty: WasmType,
    pub init: Vec<WatInstruction>,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub enum TypeDefinition {
    Rec(Vec<(String, WasmType)>),
    Type(String, WasmType),
}

impl fmt::Display for TypeDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeDefinition::Rec(types) => {
                writeln!(f, "(rec")?;
                for (name, ty) in types {
                    writeln!(f, "(type {name} {ty})")?;
                }
                writeln!(f, ")")
            }
            TypeDefinition::Type(name, ty) => writeln!(f, "(type {name} {ty})"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WatModule {
    pub tags: IndexMap<String, String>,
    pub types: Vec<TypeDefinition>,
    pub imports: Vec<(String, String, WasmType)>,
    functions: SlotMap<FunctionKey, WatFunction>,
    function_keys: Vec<FunctionKey>,
    // TODO: changet it to a struct
    pub exports: Vec<(String, String, String)>,
    pub globals: HashMap<String, Global>,
    pub data: Vec<(usize, String)>,
    pub data_offset: usize,
    pub data_offsets: HashMap<String, usize>,
    pub memories: HashMap<String, (i32, Option<i32>)>,
}

impl WatModule {
    pub fn new() -> Self {
        WatModule {
            tags: IndexMap::new(),
            types: Vec::new(),
            imports: Vec::new(),
            functions: SlotMap::with_key(),
            exports: Vec::new(),
            globals: HashMap::new(),
            data: Vec::new(),
            data_offset: 100,
            data_offsets: HashMap::new(),
            memories: HashMap::new(),
            function_keys: Default::default(),
        }
    }

    pub fn function_keys(&self) -> Vec<FunctionKey> {
        self.function_keys.clone()
    }

    pub fn add_function(&mut self, function: WatFunction) -> FunctionKey {
        let key = self.functions.insert(function);
        self.function_keys.push(key);
        key
    }

    pub fn functions_mut(&mut self) -> Vec<&mut WatFunction> {
        self.functions.values_mut().collect()
    }

    pub fn functions(&self) -> Vec<&WatFunction> {
        self.functions.values().collect()
    }

    pub fn get_function_by_key_unchecked(&self, key: FunctionKey) -> &WatFunction {
        &self.functions[key]
    }

    pub fn get_function_by_key_unchecked_mut(&mut self, key: FunctionKey) -> &mut WatFunction {
        match self.functions.get_mut(key) {
            Some(r) => r,
            None => panic!("invalid SlotMap key used"),
        }
    }

    pub fn get_function_key(&self, name: &str) -> Option<FunctionKey> {
        let name = if !name.starts_with("$") {
            format!("${name}")
        } else {
            name.to_string()
        };

        self.function_keys().into_iter().find(|k| {
            self.functions
                .get(*k)
                .map(|f| format!("${}", f.name) == name)
                .unwrap_or(false)
        })
    }

    pub fn get_function(&self, name: &str) -> Option<&WatFunction> {
        let name = if !name.starts_with("$") {
            format!("${name}")
        } else {
            name.to_string()
        };

        self.functions()
            .into_iter()
            .find(|f| format!("${}", f.name) == name)
    }

    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut WatFunction> {
        let name = if !name.starts_with("$") {
            format!("${name}")
        } else {
            name.to_string()
        };

        self.functions_mut()
            .into_iter()
            .find(|f| format!("${}", f.name) == name)
    }

    pub fn add_type(&mut self, name: impl Into<String>, ty: WasmType) {
        self.types.push(TypeDefinition::Type(name.into(), ty));
    }

    pub fn add_data(&mut self, content: String) -> (usize, usize) {
        let len = content.len();
        let offset = self.data_offset;
        if let Some(offset) = self.data_offsets.get(&content) {
            (*offset, len)
        } else {
            self._add_data_raw(offset, content)
        }
    }

    pub fn _add_data_raw(&mut self, offset: usize, content: String) -> (usize, usize) {
        let len = content.len();
        self.data.push((offset, content.clone()));
        self.data_offsets.insert(content.clone(), offset);
        let increment = if len % 4 == 0 {
            len
        } else {
            // some runtimes expect all data aligned to 4 bytes
            len + (4 - len % 4)
        };
        self.data_offset += increment;

        (offset, len)
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

    pub fn get_type_by_name(&self, search_name: &str) -> Option<WasmType> {
        for ty in &self.types {
            match ty {
                TypeDefinition::Rec(types) => {
                    for (name, ty) in types {
                        if search_name == name {
                            return Some(ty.clone());
                        }
                    }
                }
                TypeDefinition::Type(name, ty) => {
                    if search_name == name {
                        return Some(ty.clone());
                    }
                }
            }
        }

        None
    }

    pub fn append(&mut self, other: &mut Self) {
        self.types.append(&mut other.types);
        self.tags.append(&mut other.tags);
        self.imports.append(&mut other.imports);
        for function in other.functions() {
            self.add_function(function.clone());
        }
        self.exports.append(&mut other.exports);
    }

    pub fn cursor<'a>(&'a mut self) -> cursor::InstructionsCursor<'a> {
        cursor::InstructionsCursor::new(self)
    }
}

impl fmt::Display for WatModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(module")?;
        // Imports
        for (module, name, type_) in &self.imports {
            let type_str = match type_ {
                WasmType::Func { name, signature } => {
                    let params_str = signature
                        .params
                        .iter()
                        .map(|(_, ty)| format!("(param {ty})"))
                        .collect::<Vec<String>>()
                        .join(" ");
                    let result_option = signature
                        .result
                        .as_ref()
                        .map(|result| format!("(result {result})"));
                    let signature_str = vec![name.clone(), Some(params_str), result_option]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<String>>()
                        .join(" ");
                    format!("(func {signature_str})")
                }
                _ => type_.to_string(),
            };
            writeln!(f, "  (import \"{}\" \"{}\" {})", module, name, type_str)?;
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

        // Tags
        for (label, typeidx) in &self.tags {
            writeln!(f, "  (tag {label} (type {typeidx}))")?;
        }

        // Types
        for ty in &self.types {
            writeln!(f, "  {ty}")?;
        }

        // Data
        for (offset, data) in &self.data {
            write!(f, "  (data (i32.const {}) \"", offset)?;
            // TODO: this escaping should be done when inserting the data
            for &byte in data.as_bytes() {
                match byte {
                    b'"' => write!(f, "\\\"")?,
                    b'\\' => write!(f, "\\")?,
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
        for (name, global) in &self.globals {
            let ty = &global.ty;
            let init_instructions = global
                .init
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(" ");

            let ty_str = if global.mutable {
                format!("(mut {ty})")
            } else {
                format!("{ty}")
            };
            writeln!(f, "  (global {name} {ty_str} {init_instructions})")?;
        }

        // Function declarations
        for function in &self.functions() {
            writeln!(f, "  (elem declare func ${})", function.name)?;
        }

        // Functions
        for function in &self.functions() {
            write!(f, "{}", indent_str(function.to_string()))?;
        }

        // Exports
        for (name, export_type, internal_name) in &self.exports {
            writeln!(f, "  (export \"{name}\" ({export_type} {internal_name}))",)?;
        }

        writeln!(f, ")")?;

        Ok(())
    }
}

fn memory_op(f: &mut Formatter<'_>, instr: &str, label: &Option<String>) -> fmt::Result {
    if let Some(label) = label {
        write!(f, "{instr} {label}")
    } else {
        writeln!(f, "{instr}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_function_mut() {
        let mut module = WatModule::new();
        let function = WatFunction::new("foo");
        module.add_function(function);

        let function = module.get_function_mut("foo").unwrap();
        function.set_body(vec![WatInstruction::call("$bar")]);

        let function = module.get_function("foo");

        assert_eq!(
            function.unwrap().body.borrow()[0],
            WatInstruction::call("$bar")
        );
    }
}
