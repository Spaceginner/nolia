use either::Either;

#[derive(Debug, Clone)]
pub enum ItemRoot<'s> {
    CrateRoot,
    LibRoot {
        lib: &'s str,
    },
    ModRoot {
        updepth: usize,
    },
}

#[derive(Debug, Clone)]
pub struct Item<'s> {
    pub root: Option<ItemRoot<'s>>,
    pub path: Vec<&'s str>,
}

#[derive(Debug, Clone)]
pub enum IntegerTypeSize {
    Byte,   // 8
    Word,   // 16
    DWord,  // 32
    QWord,  // 64
    OWord,  // 128
}

#[derive(Debug, Clone)]
pub enum PrimitiveType {
    String,
    Char,
    Bool,
    Integer {
        signed: bool,
        size: IntegerTypeSize,
    },
    Void,
}

#[derive(Debug, Clone)]
pub enum Type<'s> {
    Op(Vec<ProtocolType<'s>>),
    Data(Item<'s>),
    Primitive(PrimitiveType),
    Array(Box<Type<'s>>),
    Never
}

#[derive(Debug, Clone)]
pub struct ProtocolType<'s> {
    pub base: Item<'s>,
    pub generics: Vec<Type<'s>>,
}

#[derive(Debug, Clone)]
pub struct Capture<'s> {
    pub name: &'s str,
    pub r#type: Type<'s>,
}

#[derive(Debug, Clone)]
pub struct GenericDef<'s> {
    pub name: &'s str,
    pub constraint: Option<Type<'s>>,
}

#[derive(Debug, Clone)]
pub struct GenericDefs<'s> {
    pub defs: Vec<GenericDef<'s>>,
}

#[derive(Debug, Clone)]
pub struct Signature<'s> {
    pub captures: Vec<Capture<'s>>,
    pub r#return: Option<Type<'s>>,
    pub generics: GenericDefs<'s>,
    pub errors: Vec<Item<'s>>,
}

#[derive(Debug, Clone)]
pub struct BlockExpression<'s> {
    pub label: Option<&'s str>,
    pub kind: Box<BlockExpressionKind<'s>>,
}

#[derive(Debug, Clone)]
pub enum BlockExpressionKind<'s> {
    Simple {
        code: StatementBlock<'s>,
    },

    Condition {
        check: Expression<'s>,
        code: StatementBlock<'s>,
        otherwise: Option<StatementBlock<'s>>,
    },
    Selector {
        of: Expression<'s>,
        cases: Vec<(Expression<'s>, StatementBlock<'s>)>,
        fallback: Option<StatementBlock<'s>>,
    },
    Handle {
        of: Expression<'s>,
        handlers: Vec<(Item<'s>, &'s str, StatementBlock<'s>)>,
        fallback: Option<(&'s str, StatementBlock<'s>)>
    },
    Unhandle {
        code: StatementBlock<'s>,
    },
    
    Loop { code: StatementBlock<'s> },
    While {
        code: StatementBlock<'s>,
        check: Expression<'s>,
        do_first: bool,
    },
    Over {
        code: StatementBlock<'s>,
        what: Expression<'s>,
        with: &'s str,
    }
}

#[derive(Debug, Clone)]
pub enum LiteralInteger {
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
pub enum LiteralExpression {
    Integer(LiteralInteger),
    Float(f64),
    Char(char),
    String(String),
    Bool(bool),
    Void,
}

#[derive(Debug, Clone)]
pub enum ConstructExpression<'s> {
    Array { vals: Vec<Expression<'s>> },
    Data {
        what: Item<'s>,
        fields: Vec<(&'s str, Expression<'s>)>,
    }
}

#[derive(Debug, Clone)]
pub enum ActionExpression<'s> {
    Call {
        what: Expression<'s>,
        args: Vec<Expression<'s>>,
    },
    MethodCall {
        what: Expression<'s>,
        method: &'s str,
        args: Vec<Expression<'s>>,
    },
    Access {
        of: Expression<'s>,
        field: &'s str,
    },
    Load {
        item: Item<'s>,
    },
}

#[derive(Debug, Clone)]
pub enum Expression<'s> {
    Action(Box<ActionExpression<'s>>),
    Block(BlockExpression<'s>),
    Construct(ConstructExpression<'s>),
    Literal(LiteralExpression),
}


impl<'s> From<StatementBlock<'s>> for Expression<'s> {
    fn from(stmt_b: StatementBlock<'s>) -> Self {
        Self::Block(BlockExpression {
            label: None,
            kind: Box::new(BlockExpressionKind::Simple {
                code: stmt_b 
            })
        })
    }
}


#[derive(Debug, Clone)]
pub enum Statement<'s> {
    Eval {
        expr: Expression<'s>,
    },
    Assignment {
        what: Expression<'s>,
        to: Expression<'s>,
    },
    Declaration {
        what: Capture<'s>,
        with: Option<Expression<'s>>,
    },
    Escape {
        value: Option<Expression<'s>>,
        label: Option<&'s str>,
    },
    Repeat {
        label: Option<&'s str>,
    },
    Return {
        value: Option<Expression<'s>>,
    },
    Throw {
        error: Expression<'s>,
    },
    VmDebug {
        dcode: u32,
    },
}

#[derive(Debug, Clone)]
pub struct StatementBlock<'s> {
    pub code: Vec<Statement<'s>>,
    pub closed: bool,
}

#[derive(Debug, Clone)]
pub struct AsmId {
    pub space: u32,
    pub item: u32,
}

#[derive(Debug, Clone)]
pub enum AsmOp<'s> {
    Pack { r#type: Either<(AsmId, usize), Item<'s>> },
    LoadConstItem { item: Either<AsmId, LiteralExpression> },
    LoadFunction { func: Either<AsmId, Item<'s>> },
    LoadImplementation { of: Either<(AsmId, u32), Item<'s>> },
    LoadSystemItem { id: Either<AsmId, &'s str> },
    Access { id: Either<u32, Item<'s>> },
    GetType,
    Call { which: usize },
    SystemCall { id: Either<AsmId, &'s str> },
    Return,
    Swap { with: usize },
    Pull { which: usize },
    Pop { count: usize, offset: usize },
    Copy { count: usize, offset: usize },
    Jump { to: Either<usize, &'s str>, check: Option<bool> },
}

#[derive(Debug, Clone)]
pub struct AsmInstruction<'s> {
    pub label: Option<&'s str>,
    pub op: AsmOp<'s>,
}

#[derive(Debug, Clone)]
pub struct AsmBlock<'s> {
    pub instrs: Vec<AsmInstruction<'s>>,
}

#[derive(Debug, Clone)]
pub struct Closure<'s> {
    pub sig: Signature<'s>,
    pub code: Either<StatementBlock<'s>, AsmBlock<'s>>,
}

#[derive(Debug, Clone)]
pub enum Declaration<'s> {
    Include {
        item: Item<'s>
    },
    Const {
        what: Capture<'s>,
        val: LiteralExpression,
    },
    Data {
        name: &'s str,
        fields: Vec<(&'s str, Type<'s>)>,
    },
    Protocol {
        name: &'s str,
        generics: GenericDefs<'s>,
        extends: Vec<Item<'s>>,
        sigs: Vec<(&'s str, Signature<'s>)>
    },
    Implementation {
        of: ProtocolType<'s>,
        r#for: Option<Item<'s>>,
        fncs: Vec<(&'s str, Closure<'s>)>,
    },
    Function {
        of: Option<Item<'s>>,
        r#impl: Option<ProtocolType<'s>>,
        name: &'s str,
        closure: Closure<'s>,
    },
}

#[derive(Debug, Clone)]
pub struct Module<'s> {
    pub decls: Vec<Declaration<'s>>
}
