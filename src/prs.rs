use regex::Regex;

pub fn expect_name<'a>(input: &'a str, name_regex: &str) -> Result<&'a str, ParseError> {
    matches_fully(input, name_regex)
        .ok_or_else(|| ParseError::WrongName { expected: name_regex.into(), found: input.into() })
}

pub fn matches_fully<'a>(input: &'a str, regex: &str) -> Option<&'a str> {
    let full_string_regex = format!("^{}$", regex);
    let regexp = Regex::new(&full_string_regex).unwrap();
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

/* ParseError */

type FailedAlternatives = Vec<(&'static str, ParseError)>;

#[derive(Debug)]
pub enum ParseError {
    WrongName { expected: String, found: String },
    WrongNumberOfArguments { expected: usize, found: usize },
    AllFailed(FailedAlternatives),
    Other(String)
}

impl From<String> for ParseError {
    fn from(value: String) -> Self {
        ParseError::Other(value)
    }
}

pub struct ParseAlternatives<'a, T> {
    input: &'a str,
    res:   Result<T, FailedAlternatives>
}

impl<'a, T> ParseAlternatives<'a, T> {
    pub fn new(s: &'a str) -> Self {
        ParseAlternatives {
            input: s,
            res:   Err(Vec::new())
        }
    }

    pub fn then_try<F: FnOnce(&'a str) -> Result<T, ParseError>>(self, name: &'static str, prs_fun: F) -> Self {
        match self.res {
            Ok(_) => self,
            Err(_) => {
                match prs_fun(self.input) {
                    Ok(x) => self.into_success(x),
                    Err(e) => self.stack_error(name, e),
                }
            }
        }
    }

    fn into_success(mut self, x: T) -> Self {
        assert!(self.res.is_err());
        self.res = Ok(x);
        self
    }

    fn stack_error(mut self, name: &'static str, e: ParseError) -> Self {
        assert!(self.res.is_err());

        // Note: we use unsafe to avoid requiring that T implements Debug
        let error_stack = unsafe {
            self.res
                .as_mut()
                .unwrap_err_unchecked()
        };

        error_stack.push((name, e));
        self
    }

    pub fn end(self) -> Result<T, ParseError> {
        self.res
            .map_err(|v| ParseError::AllFailed(v))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fully0() {
        assert_eq!(matches_fully("", "a"), None)
    }

    #[test]
    fn fully1a() {
        assert_eq!(matches_fully("a", "a"), Some("a"))
    }

    #[test]
    fn fully1b() {
        assert_eq!(matches_fully("a", "b"), None)
    }

    #[test]
    fn fully2() {
        assert_eq!(matches_fully("aa", "a"), None)
    }

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
