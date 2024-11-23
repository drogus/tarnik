use std::collections::HashMap;
use std::fmt::{self, write};
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum WasmType {
    I32,
    I64,
    F32,
    F64,
    I31Ref,
    Anyref,
    Ref(String, Nullable),
    Array { mutable: bool, ty: Box<WasmType> },
    Struct(Vec<StructField>),
}

impl fmt::Display for WasmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmType::I32 => write!(f, "i32"),
            WasmType::I64 => write!(f, "i64"),
            WasmType::F32 => write!(f, "f32"),
            WasmType::F64 => write!(f, "f64"),
            WasmType::Anyref => write!(f, "anyref"),
            WasmType::I31Ref => write!(f, "i31ref"),
            WasmType::Ref(name, nullable) => write!(f, "(ref {nullable} {name})"),
            WasmType::Array { mutable, ty } => {
                let m = if *mutable { "mut" } else { "" };
                write!(f, "(array {m} {ty})")
            }
            WasmType::Struct(fields) => {
                write!(f, "(struct")?;
                for field in fields {
                    write!(f, "  {field}")?;
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
            "i31ref" => Self::I31Ref,
            "anyref" => Self::Anyref,
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
    I32Eqz,
    I64Eqz,
    F32Eqz,
    F64Eqz,
    I32GeS,
    StructNew(String),
    ArrayNew(String),
    ArrayNewFixed(String, u16),
    ArrayLen,
    ArrayGet(String),
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
        condition: Option<Box<WatInstruction>>,
        then: Vec<Box<WatInstruction>>,
        r#else: Option<Vec<Box<WatInstruction>>>,
    },
    BrIf(String),
    Br(String),
    Empty,
    Log,
    Identifier(String),
    Drop,
    LocalTee(String),
    RefI31(Box<WatInstruction>),
    Throw(String),
    Try {
        try_block: Box<WatInstruction>,
        catches: Vec<Box<WatInstruction>>,
        catch_all: Option<Box<WatInstruction>>,
    },
    Catch(String, Box<WatInstruction>),
    CatchAll(Box<WatInstruction>),
    I32Add,
    I64Add,
    F32Add,
    F64Add,
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

    pub fn struct_new(name: impl Into<String>) -> Box<Self> {
        Box::new(Self::StructNew(name.into()))
    }

    pub fn array_new(
        name: impl Into<String>,
        init: Box<WatInstruction>,
        length: Box<WatInstruction>,
    ) -> Box<Self> {
        Box::new(Self::ArrayNew(name.into()))
    }

    pub fn array_get(name: impl Into<String>) -> Self {
        Self::ArrayGet(name.into())
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

    pub fn r#if(
        condition: Option<Box<WatInstruction>>,
        then: Vec<Box<WatInstruction>>,
        r#else: Option<Vec<Box<WatInstruction>>>,
    ) -> Box<Self> {
        Box::new(Self::If {
            condition,
            then,
            r#else,
        })
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

    pub fn ref_i31(instruction: Box<WatInstruction>) -> Box<Self> {
        Box::new(Self::RefI31(instruction))
    }

    pub fn throw(label: impl Into<String>) -> Box<Self> {
        Box::new(Self::Throw(label.into()))
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
            WatInstruction::I32Add => write!(f, "(i32.add)"),
            WatInstruction::I64Add => write!(f, "(i64.add)"),
            WatInstruction::F32Add => write!(f, "(f32.add)"),
            WatInstruction::F64Add => write!(f, "(f64.add)"),

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
            WatInstruction::ArrayNew(name) => {
                write!(f, "(array.new {name})")
            }
            WatInstruction::ArrayLen => write!(f, "(array.len)"),
            WatInstruction::ArrayGet(ty) => write!(f, "(array.get {ty})"),
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
            WatInstruction::If {
                condition,
                then,
                r#else,
            } => {
                let condition = if let Some(c) = condition {
                    format!("{c}")
                } else {
                    "".to_string()
                };
                write!(f, "(if {} (then", condition)?;
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

            WatInstruction::RefI31(instruction) => write!(f, "(ref.i31 {instruction})"),
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
    pub types: IndexMap<String, WasmType>,
    pub imports: Vec<(String, String, WasmType)>,
    pub functions: Vec<WatFunction>,
    pub exports: Vec<(String, String)>,
    pub globals: Vec<(String, WasmType, WatInstruction)>,
}

impl WatModule {
    pub fn new() -> Self {
        WatModule {
            types: IndexMap::new(),
            imports: Vec::new(),
            functions: Vec::new(),
            exports: Vec::new(),
            globals: Vec::new(),
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
}

impl fmt::Display for WatModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Types
        for (name, ty) in &self.types {
            writeln!(f, "  (type {name} {ty})")?;
        }

        // Imports
        for (module, name, type_) in &self.imports {
            writeln!(f, "  (import \"{}\" \"{}\" {})", module, name, type_)?;
        }

        // Function declarations
        for function in &self.functions {
            write!(f, "(elem declare func ${})\n", function.name)?;
        }

        // Functions
        for function in &self.functions {
            write!(f, "  {}", function)?;
        }

        // Exports
        for (name, internal_name) in &self.exports {
            writeln!(f, "  (export \"{}\" {})", name, internal_name)?;
        }

        // Globals
        for (name, type_, init) in &self.globals {
            writeln!(f, "  (global ${} {} {})", name, type_, init)?;
        }

        Ok(())
    }
}
