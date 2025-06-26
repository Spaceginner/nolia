fn main() {
    lalrpop::Configuration::default()
        .emit_rerun_directives(true)
        .set_out_dir(std::env::var("OUT_DIR").unwrap())
        .process_file("./src/parser/lang.lalrpop")
        .unwrap_or_else(|err| panic!("error generating parser: {err}"));
}
