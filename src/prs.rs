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

pub fn fun_call(input: &str) -> Option<(&str, Vec<&str>)> {
    let mut fun_name = None;
    let mut params = Vec::new();
    let mut paren_stack = 0;
    let mut last_pos = 0;

    for (new_pos, c) in input.chars()
                                .enumerate()
                                .filter(|(_, c)|
                                    *c == '(' || *c == ')' || *c == ',')
    {
        match c {
            '(' => {
                if paren_stack > 0 {
                    // We are already parsing the arguments
                    // We will now ignore everything until the correct closing paren
                    paren_stack += 1;
                    // Important: don't update last_pos here
                }
                else {
                    // First opening parenthesis
                    // Keep track of the fun_name
                    if fun_name.is_some() || last_pos != 0 {
                        return None;
                    }
                    assert!(params.len() == 0);
                    if new_pos == 0 {
                        return None;
                    }
                    fun_name = Some(&input[0..new_pos]);
                    paren_stack = 1;
                    last_pos = new_pos + 1;  // ignore the parenthesis itself
                }
            }
            ')' => {
                if paren_stack <= 0 {
                    return None;
                }
                if paren_stack == 1 {
                    if new_pos != (input.len() - 1) {
                        return None;
                    }
                    else {
                        // This is the last argument
                        assert!(fun_name.is_some());
                        // Note: here we accept a 0-length argument to allow inputs of the form "a()"
                        if new_pos == last_pos {
                            if params.len() > 0 {
                                return None;
                            }
                        }
                        else {
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
                        // Invalid input
                        return None;
                    }
                    1 => {
                        // We now have an argument
                        assert!(fun_name.is_some());
                        if last_pos == new_pos {
                            return None;
                        }
                        params.push(&input[last_pos..new_pos]);
                        last_pos = new_pos + 1;
                    }
                    _ => {
                        assert!(paren_stack > 1);
                        // Nothing to do here
                    }
                }
            }
            _   => unreachable!(),
        }
    }

    if paren_stack != 0 {
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
