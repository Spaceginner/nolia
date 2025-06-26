use super::lcr::Path;

#[macro_export]
macro_rules! path {
    () => {
        Path { crate_: None, parts: vec![] }
    };
    ($part:ident $(:: $parts:ident)*) => {
        Path { crate_: None, parts: vec![stringify!($part).into(), $(stringify!($parts).into()),*] }
    };
    ($crate_:ident @ $part:ident $(:: $parts:ident)*) => {
        Path { crate_: Some(stringify!($crate_).into()), parts: vec![stringify!($part).into(), $(stringify!($parts).into()),*] }
    };
}

impl Path {
    pub fn extend(mut self, part: &str) -> Self {
        self.parts.push(part.into());
        self
    }

    pub fn combine(self, path: Vec<&str>) -> Self {
        path.into_iter().fold(self, |p, i| p.extend(i))
    }

    pub fn pop(mut self, n: usize) -> Self {
        (0..n).for_each(|_| { let _ = self.parts.pop().expect("too bad; path breaks out"); });
        self
    }

    pub fn noc(mut self) -> Self {
        self.crate_ = None;
        self
    }
}
