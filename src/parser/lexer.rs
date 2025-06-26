use std::borrow::Cow;
use std::num::{ParseFloatError, ParseIntError};
use lexgen::lexer;

pub type Loc = lexgen_util::Loc;
pub type Error = lexgen_util::LexerError<CtxError>;

#[derive(Clone, Debug)]
pub enum Token<'s> {
    Symbol(Symbol),
    Keyword(Keyword),
    Identifier(&'s str),
    Literal(Literal<'s>),
}

#[derive(Clone, Debug)]
pub enum Literal<'s> {
    String(Cow<'s, str>),
    Char(char),
    Integer(Integer),
    Float(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum Integer {
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

impl std::ops::Neg for Integer {
    type Output = Result<Self, CtxError>;

    fn neg(self) -> Self::Output {
        Ok(match self {
            Self::I8(n) => Self::I8(-n),
            Self::I16(n) => Self::I16(-n),
            Self::I32(n) => Self::I32(-n),
            Self::I64(n) => Self::I64(-n),
            Self::I128(n) => Self::I128(-n),

            Self::U8(_) | Self::U16(_) | Self::U32(_)
            | Self::U64(_) | Self::U128(_) => Err(CtxError::NegativeUnsigned)?,
        })
    }
}

#[derive(Clone, Debug)]
pub enum Keyword {
    Str,
    Char,
    Bool,
    Void,
    True,
    False,
    If,
    Unless,
    While,
    Until,
    Sel,
    Or,
    Else,
    Of,
    Loop,
    Do,
    Over,
    With,
    Handle,
    Unhandle,
    Let,
    Escape,
    Repeat,
    Return,
    Throw,
    VmDebug,
    Pack,
    LoadItem,
    LoadFunc,
    LoadImpl,
    LoadSysItem,
    Access,
    Call,
    SysCall,
    Swap,
    Pull,
    Pop,
    Copy,
    Jump,
    Asm,
    Include,
    Module,
    Const,
    Data,
    Proto,
    Impl,
    Fnc,
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
}

#[derive(Clone, Debug)]
pub enum Symbol {
    Hashtag,
    PathSep,
    Dollar,
    At,
    SquareOpen,
    SquareClose,
    Plus,
    Percent,
    Exclamation,
    TwoDots,
    BracketOpen,
    BracketClose,
    Arrow,
    Line,
    Comma,
    Dot,
    CurlyOpen,
    CurlyClose,
    Equal,
    DotComma,
    Slash,
    Diamond,
    AngledOpen,
    AngledClose,
}

#[derive(Debug, Clone)]
pub enum CtxError {
    InvalidFloat(ParseFloatError),
    UnexpectedIntElement,
    IntlessInt,
    NegativeUnsigned,
    InvalidInt(ParseIntError),
}

#[derive(Debug, Default, Eq, PartialEq)]
enum IntSign {
    #[default]
    Pos,
    Neg,
}

#[derive(Debug, Default)]
enum IntBase {
    #[default]
    Decimal = 10,
    Binary = 2,
    Hex = 16,
    Octal = 8,
}

#[derive(Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
enum IntElement {
    #[default]
    // Sign,  // technically speaking, it will never be at this state, as it will always go over it already
    Base,
    Int,
    Suffix,
}

impl IntElement {
    fn next(&mut self) {
        *self = match self {
            // Self::Sign => Self::Base,
            Self::Base => Self::Int,
            Self::Int => Self::Suffix,
            Self::Suffix => unreachable!(),
        };
    }

    fn now(&mut self, el: Self) -> Result<(), CtxError> {
        if *self > el {
            Err(CtxError::UnexpectedIntElement)
        } else {
            *self = el;
            Ok(())
        }
    }
}

#[derive(Debug, Default)]
struct IntegerLexerState<'s> {
    sign: IntSign,
    base: IntBase,
    expected_el: IntElement,
    buffer: Option<&'s str>,
}

#[derive(Debug, Default)]
struct StringLexerState {
    interpreted: Option<String>,
}

fn handle_escaped<'s>(lexer: &mut Lexer<'s, impl Iterator<Item = char> + Clone>, c: char) -> lexgen_util::SemanticActionResult<Token<'s>> {
    lexer.state().str_lex.interpreted.as_mut().expect("switching into escaping creates the buffer").push(c);
    lexer.reset_match();
    lexer.switch(LexerRule::String)
}

#[derive(Debug, Default)]
pub struct State<'s> {
    int_lex: IntegerLexerState<'s>,
    str_lex: StringLexerState,
}

impl State<'_> {
    fn reset_int(&mut self) {
        self.int_lex = IntegerLexerState::default();
    }
}

lexer! {
    pub Lexer(State<'input>) -> Token<'input>;

    type Error = CtxError;

    let whitespace = [' ' '\t' '\n'] | "\r\n";

    rule Init {
        "//" => |lexer| lexer.switch(LexerRule::SingleLineComment),
        "/*" => |lexer| lexer.switch(LexerRule::MultiLineComment),

        $whitespace,

        let ident = $$XID_Start $$XID_Continue*;

        "r#" $ident => |lexer| {
            lexer.return_(Token::Identifier(&lexer.match_()[2..]))
        },

        "#" = Token::Symbol(Symbol::Hashtag),
        "::" = Token::Symbol(Symbol::PathSep),
        "$" = Token::Symbol(Symbol::Dollar),
        "@" = Token::Symbol(Symbol::At),
        "[" = Token::Symbol(Symbol::SquareOpen),
        "]" = Token::Symbol(Symbol::SquareClose),
        "+" = Token::Symbol(Symbol::Plus),
        "%" = Token::Symbol(Symbol::Percent),
        "!" = Token::Symbol(Symbol::Exclamation),
        ":" = Token::Symbol(Symbol::TwoDots),
        "(" = Token::Symbol(Symbol::BracketOpen),
        ")" = Token::Symbol(Symbol::BracketClose),
        "->" = Token::Symbol(Symbol::Arrow),
        "|" = Token::Symbol(Symbol::Line),
        "," = Token::Symbol(Symbol::Comma),
        "{" = Token::Symbol(Symbol::CurlyOpen),
        "}" = Token::Symbol(Symbol::CurlyClose),
        "." = Token::Symbol(Symbol::Dot),
        "=" = Token::Symbol(Symbol::Equal),
        ";" = Token::Symbol(Symbol::DotComma),
        "/" = Token::Symbol(Symbol::Slash),
        "<>" = Token::Symbol(Symbol::Diamond),
        "<" = Token::Symbol(Symbol::AngledOpen),
        ">" = Token::Symbol(Symbol::AngledClose),

        "str" = Token::Keyword(Keyword::Str),
        "char" = Token::Keyword(Keyword::Char),
        "bool" = Token::Keyword(Keyword::Bool),
        "void" = Token::Keyword(Keyword::Void),
        "true" = Token::Keyword(Keyword::True),
        "false" = Token::Keyword(Keyword::False),
        "if" = Token::Keyword(Keyword::If),
        "unless" = Token::Keyword(Keyword::Unless),
        "while" = Token::Keyword(Keyword::While),
        "until" = Token::Keyword(Keyword::Until),
        "sel" = Token::Keyword(Keyword::Sel),
        "or" = Token::Keyword(Keyword::Or),
        "else" = Token::Keyword(Keyword::Else),
        "of" = Token::Keyword(Keyword::Of),
        "loop" = Token::Keyword(Keyword::Loop),
        "do" = Token::Keyword(Keyword::Do),
        "over" = Token::Keyword(Keyword::Over),
        "with" = Token::Keyword(Keyword::With),
        "handle" = Token::Keyword(Keyword::Handle),
        "unhandle" = Token::Keyword(Keyword::Unhandle),
        "let" = Token::Keyword(Keyword::Let),
        "escape" = Token::Keyword(Keyword::Escape),
        "repeat" = Token::Keyword(Keyword::Repeat),
        "return" = Token::Keyword(Keyword::Return),
        "throw" = Token::Keyword(Keyword::Throw),
        "vmdebug" = Token::Keyword(Keyword::VmDebug),
        "pack" = Token::Keyword(Keyword::Pack),
        "loaditem" = Token::Keyword(Keyword::LoadItem),
        "loadfunc" = Token::Keyword(Keyword::LoadFunc),
        "loadimpl" = Token::Keyword(Keyword::LoadImpl),
        "loadsysitem" = Token::Keyword(Keyword::LoadSysItem),
        "access" = Token::Keyword(Keyword::Access),
        "call" = Token::Keyword(Keyword::Call),
        "syscall" = Token::Keyword(Keyword::SysCall),
        "return" = Token::Keyword(Keyword::Return),
        "swap" = Token::Keyword(Keyword::Swap),
        "pull" = Token::Keyword(Keyword::Pull),
        "pop" = Token::Keyword(Keyword::Pop),
        "copy" = Token::Keyword(Keyword::Copy),
        "asm" = Token::Keyword(Keyword::Asm),
        "jump" = Token::Keyword(Keyword::Jump),
        "include" = Token::Keyword(Keyword::Include),
        "module" = Token::Keyword(Keyword::Module),
        "const" = Token::Keyword(Keyword::Const),
        "data" = Token::Keyword(Keyword::Data),
        "proto" = Token::Keyword(Keyword::Proto),
        "impl" = Token::Keyword(Keyword::Impl),
        "fnc" = Token::Keyword(Keyword::Fnc),

        "i8" = Token::Keyword(Keyword::I8),
        "i16" = Token::Keyword(Keyword::I16),
        "i32" = Token::Keyword(Keyword::I32),
        "i64" = Token::Keyword(Keyword::I64),
        "i128" = Token::Keyword(Keyword::I128),
        "u8" = Token::Keyword(Keyword::U8),
        "u16" = Token::Keyword(Keyword::U16),
        "u32" = Token::Keyword(Keyword::U32),
        "u64" = Token::Keyword(Keyword::U64),
        "u128" = Token::Keyword(Keyword::U128),

        // ident
        $ident => |lexer| {
            lexer.return_(Token::Identifier(lexer.match_()))
        },

        // char
        // todo support for char escaping
        "'" => |lexer| {
            lexer.reset_match();
            lexer.switch(LexerRule::Char)
        },

        // str
        "\"" => |lexer| {
            lexer.reset_match();
            lexer.switch(LexerRule::String)
        },

        // float
        ("+" | "-")? ['0'-'9']+ "." ['0'-'9']+ ("e" ("+" | "-") ['0'-'9']+)? =? |lexer| {
            lexer.return_(try { Token::Literal(Literal::Float(lexer.match_().parse().map_err(CtxError::InvalidFloat)?)) })
        },

        // int
        "+" | "-" => |lexer| {
            lexer.state().int_lex.sign = if lexer.match_() == "+" { IntSign::Pos } else { IntSign::Neg };
            lexer.reset_match();
            lexer.switch(LexerRule::Integer)
        },

        "0d" | "0b" | "0x" | "0o" => |lexer| {
            lexer.state().int_lex.expected_el = IntElement::Int;
            lexer.state().int_lex.base = match lexer.match_() {
                "0d" => IntBase::Decimal,
                "0b" => IntBase::Binary,
                "0x" => IntBase::Hex,
                "0o" => IntBase::Octal,
                _ => unreachable!(),
            };
            lexer.reset_match();
            lexer.switch(LexerRule::Integer)
        },

        ['0'-'9']+ > (_ # ['i' 'u']) =? |lexer| {
            lexer.return_(try { Token::Literal(Literal::Integer(Integer::I64(lexer.match_().parse().map_err(CtxError::InvalidInt)?))) })
        },

        ['0'-'9']+ => |lexer| {
            lexer.state().int_lex.expected_el = IntElement::Suffix;
            lexer.reset_match();
            lexer.switch(LexerRule::Integer)
        },
    }

    rule Integer {
        "0d" | "0b" | "0x" | "0o" =? |lexer| {
            match try {
                lexer.state().int_lex.expected_el.now(IntElement::Base)?;
                lexer.state().int_lex.base = match lexer.match_() {
                    "0d" => IntBase::Decimal,
                    "0b" => IntBase::Binary,
                    "0x" => IntBase::Hex,
                    "0o" => IntBase::Octal,
                    _ => unreachable!(),
                };
                lexer.reset_match();
            } {
                Ok(()) => lexer.continue_(),
                Err(err) => lexer.return_(Err(err)),
            }
        },

        (['0'-'9' 'a'-'f']+ | ['0'-'9' 'A'-'F']+) > (_ # ['i' 'u']) =? |lexer| {
            let res = try {
                lexer.state().int_lex.expected_el.now(IntElement::Int)?;

                let parser = |n, r| Ok(Integer::I64(i64::from_str_radix(n, r)?));
                let raw = lexer.match_();
                let mut integer = match lexer.state().int_lex.base {
                    IntBase::Decimal => parser(raw, 10),
                    IntBase::Binary => parser(raw, 2),
                    IntBase::Hex => parser(raw, 16),
                    IntBase::Octal => parser(raw, 8),
                }.map_err(CtxError::InvalidInt)?;

                if lexer.state().int_lex.sign == IntSign::Neg {
                    integer = (-integer)?;
                };

                lexer.state().reset_int();
                Token::Literal(Literal::Integer(integer))
            };
            lexer.switch_and_return(LexerRule::Init, res)
        },

        ['0'-'9' 'a'-'f']+ | ['0'-'9' 'A'-'F']+ =? |lexer| {
            match try {
                lexer.state().int_lex.expected_el.now(IntElement::Int)?;
                lexer.state().int_lex.buffer = Some(lexer.match_());
                lexer.reset_match();
            } {
                Ok(()) => lexer.continue_(),
                Err(err) => lexer.return_(Err(err)),
            }
        },

        "i8" | "i16" | "i32" | "i64" | "i128"
            | "u8" | "u16" | "u32" | "u64" | "u128" =? |lexer| {
            let res = try {
                lexer.state().int_lex.expected_el.now(IntElement::Suffix)?;

                let parser = match lexer.match_() {
                    "i8" => |n, r| Ok::<_, ParseIntError>(Integer::I8(i8::from_str_radix(n, r)?)),
                    "i16" => |n, r| Ok(Integer::I16(i16::from_str_radix(n, r)?)),
                    "i32" => |n, r| Ok(Integer::I32(i32::from_str_radix(n, r)?)),
                    "i64" => |n, r| Ok(Integer::I64(i64::from_str_radix(n, r)?)),
                    "i128" => |n, r| Ok(Integer::I128(i128::from_str_radix(n, r)?)),
                    "u8" => |n, r| Ok(Integer::U8(u8::from_str_radix(n, r)?)),
                    "u16" => |n, r| Ok(Integer::U16(u16::from_str_radix(n, r)?)),
                    "u32" => |n, r| Ok(Integer::U32(u32::from_str_radix(n, r)?)),
                    "u64" => |n, r| Ok(Integer::U64(u64::from_str_radix(n, r)?)),
                    "u128" => |n, r| Ok(Integer::U128(u128::from_str_radix(n, r)?)),
                    _ => unreachable!(),
                };

                let raw = lexer.state().int_lex.buffer.take().ok_or(CtxError::IntlessInt)?;
                let mut integer = match lexer.state().int_lex.base {
                    IntBase::Decimal => parser(raw, 10),
                    IntBase::Binary => parser(raw, 2),
                    IntBase::Hex => parser(raw, 16),
                    IntBase::Octal => parser(raw, 8),
                }.map_err(CtxError::InvalidInt)?;

                if lexer.state().int_lex.sign == IntSign::Neg {
                    integer = (-integer)?;
                };

                lexer.state().reset_int();
                Token::Literal(Literal::Integer(integer))
            };

            lexer.switch_and_return(LexerRule::Init, res)
        },
    }

    rule Char {
        _ => |lexer| {
            let c = lexer.match_().chars().next().expect("pattern matches specifically any one char");
            lexer.switch_and_return(LexerRule::Init, Token::Literal(Literal::Char(c)))
        },
    }

    rule String {
        "\"" => |lexer| {
            let m = lexer.match_();
            let m = &m[..m.len()-1];

            let i_s = lexer.state().str_lex.interpreted.take();
            lexer.switch_and_return(
                LexerRule::Init,
                Token::Literal(Literal::String(
                    match i_s {
                        None => Cow::Borrowed(m),
                        Some(mut s) => {
                            s.push_str(m);
                            Cow::Owned(s)
                        },
                    }
                ))
            )
        },

        "\\" => |lexer| {
            let m = lexer.match_();
            let m = &m[..m.len()-1];

            let i_s = &mut lexer.state().str_lex.interpreted;
            match i_s {
                None => {
                    *i_s = Some(m.into());
                },
                Some(s) => {
                    s.push_str(m);
                },
            };

            lexer.switch(LexerRule::StringEscape)
        },

        _ => |lexer| lexer.continue_(),
    }

    rule StringEscape {
        "\\" => |lexer| handle_escaped(lexer, '\\'),
        "\"" => |lexer| handle_escaped(lexer, '"'),
        "n" => |lexer| handle_escaped(lexer, '\n'),
        "r" => |lexer| handle_escaped(lexer, '\r'),
        "t" => |lexer| handle_escaped(lexer, '\t'),
        "0" => |lexer| handle_escaped(lexer, '\0'),
        "b" => |lexer| handle_escaped(lexer, '\x08'),
        "e" => |lexer| handle_escaped(lexer, '\x1b'),
        "a" => |lexer| handle_escaped(lexer, '\x0b'),
    }

    rule SingleLineComment {
        "\n" => |lexer| { lexer.reset_match(); lexer.switch(LexerRule::Init) },
        _,
    }

    rule MultiLineComment {
        "*/" => |lexer| { lexer.reset_match(); lexer.switch(LexerRule::Init) },
        _,
    }
}
