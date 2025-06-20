use either::Either;
use std::num::NonZero;
use std::collections::HashMap;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Path {
    pub crate_: Option<Box<str>>,
    pub parts: Vec<Box<str>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub (in super::super) enum IntegerTypeSize {
    Byte,   // 8
    Word,   // 16
    DWord,  // 32
    QWord,  // 64
    OWord,  // 128
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub (in super::super) enum PrimitiveType {
    String,
    Char,
    Bool,
    Integer {
        signed: bool,
        size: IntegerTypeSize,
    },
    Float,
    Void,
}

#[derive(Debug, Clone)]
pub (in super::super) struct DataType {
    pub fields: Vec<(Box<str>, TypeRef)>
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub (in super::super) struct ConcreteTypeId {
    pub path: Path,
}


#[derive(Debug, Clone)]
pub (in super::super) struct ConcreteTypeRef {
    pub id: ConcreteTypeId,
    pub generics: Vec<TypeRef>,
}


#[derive(Debug, Clone)]
pub (in super::super) enum TypeRef {
    Op(Vec<ConcreteTypeRef>),
    Data(ConcreteTypeId),
    Primitive(PrimitiveType),
    Array(Option<Box<TypeRef>>),
    Function(Box<FunctionType>),
    Nothing,
    Never,
}

#[derive(Debug, Clone)]
pub (in super::super) struct FunctionType {
    pub captures: Vec<(Box<str>, TypeRef)>,
    pub r#return: TypeRef,
    pub generics: GenericDefs,
    pub errors: Vec<ConcreteTypeId>,
}

pub (in super::super) type GenericDefs = Vec<(Box<str>, Option<TypeRef>)>;

#[derive(Debug, Clone)]
pub (in super::super) struct ProtocolType {
    pub generics: GenericDefs,
    pub extends: Vec<ConcreteTypeId>,
    pub sigs: Vec<(Box<str>, FunctionType)>
}

#[derive(Debug, Clone)]
pub (in super::super) struct Declaration {
    pub name: Box<str>,
    pub r#type: TypeRef,
}

#[derive(Debug, Clone)]
pub (in super::super) struct IntermediatePath {
    pub var: Option<Box<str>>,
    pub path: Path,
}

#[derive(Debug, Clone)]
pub (in super::super) enum ActionInstruction {
    Load {
        item: IntermediatePath,
    },
    Call {
        what: SBlock,
        args: Vec<SBlock>,
    },
    Access {
        of: SBlock,
        field: Box<str>,
    },
    MethodCall {
        what: SBlock,
        method: Box<str>,
        args: Vec<SBlock>,
    },
}

#[derive(Debug, Clone)]
pub (in super::super) enum LiteralInteger {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
}


#[derive(Debug, Clone)]
pub (in super::super) enum LiteralValue {
    Integer(LiteralInteger),
    Float(f64),
    Char(char),
    String(Box<str>),
    Bool(bool),
    Void,
}

#[derive(Debug, Clone)]
pub (in super::super) enum ConstructInstruction {
    Array {
        vals: Vec<SBlock>,
    },
    Data {
        what: ConcreteTypeId,
        fields: Vec<(Box<str>, SBlock)>
    },
}

#[derive(Debug, Clone)]
pub (in super::super) enum StatementInstruction {
    Assignment {
        what: SBlock,
        to: SBlock,
    },
    Repeat {
        label: Box<str>,
    },
    Escape {
        value: Option<SBlock>,
        label: Box<str>,
    },
    Return {
        value: Option<SBlock>,
    },
    Throw {
        error: SBlock,
    },
    VmDebug  {
        dcode: u32,
    },
}

#[derive(Debug, Clone)]
pub (in super::super) enum Instruction {
    DoBlock(SBlock),
    LoadLiteral(LiteralValue),
    DoAction(ActionInstruction),
    Construct(ConstructInstruction),
    DoStatement(StatementInstruction),
}


#[derive(Debug, Clone)]
pub (in super::super) enum SBlockTag {
    // used internally when desugaring to not completely spell out simple block each time just to put a label
    Block {
        block: SBlock,
    },
    Simple {
        decls: Vec<Declaration>,
        code: Vec<Instruction>,
        closed: bool,
    },
    // todo later merge with selector
    Condition {
        check: SBlock,
        code: SBlock,
        otherwise: Option<SBlock>,
        inverted: bool,
    },
    Selector {
        of: SBlock,
        cases: Vec<(SBlock, SBlock)>,
        fallback: Option<SBlock>,
    },
    Handle {
        what: SBlock,
        handlers: Vec<(ConcreteTypeId, Box<str>, SBlock)>,
        fallback: Option<(Box<str>, SBlock)>,
    },
    // todo later merge with handle
    Unhandle {
        what: SBlock,
    },
    // todo later merge with simple block
    Over {
        code: SBlock,
        what: SBlock,
        with: Box<str>,
    }
}

#[derive(Debug, Clone)]
pub (in super::super) struct SBlock {
    pub tag: Box<SBlockTag>,
    pub label: Option<Box<str>>,
}

#[derive(Debug, Clone)]
pub (in super::super) struct AsmId {
    pub space: Option<NonZero<u32>>,
    pub item: u32,
}


#[derive(Debug, Clone)]
pub (in super::super) enum AsmOp {
    Pack { r#type: Either<(AsmId, usize), ConcreteTypeId> },
    LoadConstItem { item: Either<AsmId, LiteralValue> },
    LoadFunction { func: Either<AsmId, Path> },
    LoadImplementation { of: Either<(AsmId, u32), Path> },
    LoadSystemItem { id: Either<AsmId, Box<str>> },
    Access { id: Either<u32, Path> },
    Call { which: usize },
    SystemCall { id: Either<AsmId, Box<str>> },
    Return,
    Swap { with: usize },
    Pull { which: usize },
    Pop { count: usize, offset: usize },
    Copy { count: usize, offset: usize },
    Jump { to: Either<usize, Box<str>>, check: Option<bool> },
}

#[derive(Debug, Clone)]
pub (in super::super) struct AsmInstruction {
    pub op: AsmOp,
    pub label: Option<Box<str>>,
}

#[derive(Debug, Clone)]
pub (in super::super) struct AsmBlock {
    pub code: Vec<AsmInstruction>,
}

#[derive(Debug, Clone)]
pub (in super::super) enum Block {
    Structured(SBlock),
    Asm(AsmBlock),
}

#[derive(Debug, Clone)]
pub (in super::super) struct Function {
    pub r#type: FunctionType,
    pub code: Block,
}


#[derive(Debug, Clone)]
pub (in super::super) struct Crate {
    pub deps: Vec<(Box<str>, (u16, u16, u16))>,
    pub implementation_store: HashMap<ConcreteTypeId, Vec<(ConcreteTypeId, Vec<Function>)>>,  // data type id -> (protocol id, impl funcs)
    pub function_store: HashMap<Path, Function>,
    pub method_store: HashMap<TypeRef, Vec<Function>>,
    pub item_store: HashMap<Path, LiteralValue>,
}
