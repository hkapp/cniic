use regex::Regex;

pub fn matches_fully<'a>(input: &'a str, regex: &str) -> Option<&'a str> {
    let regexp = Regex::new(regex).unwrap();
    let matches = regexp.captures(input)?;

    // We expect one capture group for the entire match
    if matches.len() == 1 {
        matches.get(0)
            .map(|m| m.as_str())
    }
    else {
        None
    }
}

/// Parse a function call of the form `<fun-name>([<arg>,]...)`
pub fn fun_call(input: &str) -> Option<(&str, Vec<&str>)> {
    let mut fun_name = None;
    let mut params = Vec::new();
    let mut paren_stack: u16 = 0;
    let mut last_pos: usize = 0;

    for (new_pos, c) in input.chars()
                                .enumerate()
                                .filter(|(_, c)|
                                    *c == '(' || *c == ')' || *c == ',')
    {
        match c {
            '(' => {
                if paren_stack == 0 {
                    if fun_name.is_some() || last_pos != 0 {
                        // Invalid input:
                        // a()(b, c)
                        //    ^
                        return None;
                    }
                    assert!(params.len() == 0);
                    if new_pos == 0 {
                        // Invalid input:
                        // (b, c)
                        // ^
                        return None;
                    }

                    // The input is valid:
                    // a(b, c)
                    //  ^
                    // This is the first opening parenthesis
                    // Keep track of the fun_name
                    fun_name = Some(&input[0..new_pos]);
                    paren_stack = 1;
                    last_pos = new_pos + 1;  // ignore the parenthesis itself
                }
                else {
                    // We are already parsing the arguments
                    // a(b, c(d, e))
                    //       ^
                    // We now ignore everything until the corresponding closing paren
                    paren_stack += 1;
                    // Important: don't update last_pos here
                }
            }
            ')' => {
                if paren_stack == 0 {
                    // Invalid input:
                    // a())
                    //    ^
                    return None;
                }
                if paren_stack == 1 {
                    if new_pos != (input.len() - 1) {
                        // Invalid input:
                        // a(b, c)d
                        //       ^
                        return None;
                    }
                    else {
                        // This is the last argument
                        assert!(fun_name.is_some());

                        if new_pos == last_pos {
                            if params.len() > 0 {
                                // Invalid input:
                                // a(b, c,)
                                //        ^
                                return None;
                            }
                            // else: valid input
                            // a()
                            //   ^
                        }
                        else {
                            // Valid input:
                            // a(b, c)
                            //       ^
                            params.push(&input[last_pos..new_pos]);
                        }
                        last_pos = new_pos + 1;
                    }
                }
                paren_stack -= 1;
                // Important: don't update last_pos here
            }
            ',' => {
                match paren_stack {
                    0 => {
                        // Invalid input:
                        // a,b
                        //  ^
                        return None;
                    }
                    1 => {
                        // We now have an argument
                        assert!(fun_name.is_some());

                        if last_pos == new_pos {
                            // Invalid input:
                            // a(b,,c)
                            //     ^
                            return None;
                        }

                        // Valid input:
                        // a(b, c)
                        //    ^
                        params.push(&input[last_pos..new_pos]);
                        last_pos = new_pos + 1;
                    }
                    _ => {
                        assert!(paren_stack > 1);
                        // Nothing to do here:
                        // a(b, c(d, e))
                        //         ^
                    }
                }
            }
            _   => unreachable!(),
        }
    }

    if paren_stack != 0 {
        // Invalid input:
        // a(
        //  ^
        return None;
    }
    else {
        fun_name.map(|n| (n, params))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_fun_call_success(input: &str, expected_name: &str, expected_args: Vec<&str>) {
        assert_eq!(fun_call(input), Some((expected_name, expected_args)));
    }

    fn test_fun_call_failure(input: &str) {
        assert_eq!(fun_call(input), None);
    }

    #[test]
    fn fun_call1() {
        test_fun_call_failure("a");
    }

    #[test]
    fn fun_call2() {
        test_fun_call_failure("a(");
    }

    #[test]
    fn fun_call3() {
        test_fun_call_success("a()", "a", Vec::new());
    }

    #[test]
    fn fun_call4() {
        test_fun_call_failure("a(b");
    }

    #[test]
    fn fun_call5() {
        test_fun_call_success("a(b)", "a", vec!["b"]);
    }

    #[test]
    fn fun_call6() {
        test_fun_call_failure("a(b,");
    }

    #[test]
    fn fun_call7() {
        test_fun_call_failure("a(b,)");
    }

    #[test]
    fn fun_call8() {
        test_fun_call_success("a(b,c)", "a", vec!["b", "c"]);
    }

    #[test]
    fn fun_call9() {
        test_fun_call_failure("()");
    }

    #[test]
    fn fun_call10() {
        test_fun_call_failure("(a)");
    }

    #[test]
    fn fun_call11() {
        test_fun_call_failure("");
    }
}
