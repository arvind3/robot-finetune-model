# Robot Framework Advanced Patterns (Authoritative Summary)

Primary references:
- https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html
- https://docs.robotframework.org/docs

## TRY/EXCEPT/FINALLY (Robot Framework 5+)

Robot Framework 5 introduced native error handling blocks:

```robot
*** Test Cases ***
Safe Operation
    TRY
        Perform Risky Action
    EXCEPT    *timeout*    type=GLOB    AS    ${err}
        Log    Caught timeout: ${err}
    EXCEPT
        Log    Unexpected error
    FINALLY
        Log    Always runs
    END

*** Keywords ***
Perform Risky Action
    Log    attempting
```

Catch any exception and store the message:

```robot
*** Test Cases ***
Catch Any Error
    TRY
        Fail    Something went wrong
    EXCEPT    AS    ${msg}
        Log    Caught: ${msg}
    END
```

Full TRY/EXCEPT/ELSE/FINALLY:

```robot
*** Test Cases ***
Full Error Handling
    TRY
        Log    attempt
    EXCEPT
        Log    error path
    ELSE
        Log    success path
    FINALLY
        Log    always runs
    END
```

## FOR loop patterns

Iterate over a list:

```robot
*** Variables ***
@{ITEMS}    alpha    beta    gamma

*** Test Cases ***
Iterate Items
    FOR    ${item}    IN    @{ITEMS}
        Log    Processing ${item}
    END
```

Iterate with IN RANGE:

```robot
*** Test Cases ***
Iterate Range
    FOR    ${i}    IN RANGE    1    11
        Log    Step ${i}
    END
```

Break early with BREAK:

```robot
*** Test Cases ***
Find First Match
    FOR    ${item}    IN    red    green    blue
        IF    '${item}' == 'green'
            Log    Found: ${item}
            BREAK
        END
    END
```

Skip with CONTINUE:

```robot
*** Test Cases ***
Skip Odd Numbers
    FOR    ${n}    IN RANGE    1    8
        IF    ${n} % 2 != 0
            CONTINUE
        END
        Log    Even: ${n}
    END
```

Iterate over dictionary:

```robot
*** Variables ***
&{CONFIG}    host=localhost    port=8080

*** Test Cases ***
Log Config
    FOR    ${key}    ${value}    IN    &{CONFIG}
        Log    ${key}=${value}
    END
```

Nested FOR loops:

```robot
*** Test Cases ***
Grid Iteration
    FOR    ${row}    IN RANGE    1    4
        FOR    ${col}    IN RANGE    1    4
            Log    cell(${row},${col})
        END
    END
```

## WHILE loops (Robot Framework 5+)

Always set a `limit` to prevent infinite loops:

```robot
*** Test Cases ***
Poll Until Ready
    ${count}=    Set Variable    0
    WHILE    ${count} < 5    limit=20
        ${count}=    Evaluate    ${count} + 1
        Log    count=${count}
    END
```

## IF/ELSE IF/ELSE

```robot
*** Test Cases ***
Grade Check
    ${score}=    Set Variable    75
    IF    ${score} >= 90
        Log    Grade: A
    ELSE IF    ${score} >= 70
        Log    Grade: B
    ELSE
        Log    Grade: C
    END
```

## Inline IF

```robot
*** Test Cases ***
Quick Check
    ${debug}=    Set Variable    ${True}
    IF    ${debug}    Log    Debug active    level=DEBUG
```

## Skip If

```robot
*** Variables ***
${ENV}    staging

*** Test Cases ***
Prod Only Test
    Skip If    '${ENV}' != 'prod'    Not running in production.
    Log    Production-only check passed.
```

## Variable scoping

```robot
*** Keywords ***
Set Scope Examples
    # Local — visible only in this keyword
    ${local}=    Set Variable    local_value

    # Suite-level — visible to all tests in the suite
    Set Suite Variable    ${SUITE_VAR}    suite_value

    # Global — visible across all suites (use sparingly)
    Set Global Variable    ${GLOBAL_VAR}    global_value
```

## RETURN keyword (modern style)

```robot
*** Keywords ***
Compute Sum
    [Arguments]    ${a}    ${b}
    ${result}=    Evaluate    ${a} + ${b}
    RETURN    ${result}

*** Test Cases ***
Sum Test
    ${total}=    Compute Sum    3    4
    Should Be Equal As Integers    ${total}    7
```

## Keyword with default arguments

```robot
*** Keywords ***
Greet User
    [Arguments]    ${name}    ${greeting}=Hello
    Log    ${greeting}, ${name}!
```

## Resource file directory structure

```
project/
  tests/
    suite_a.robot
    suite_b.robot
  resources/
    browser.robot
    api.robot
    common.robot
  variables/
    env.py
    test_data.yaml
```

Each resource file uses `*** Settings ***` and `*** Keywords ***` but no `*** Test Cases ***`.

## Tag-driven execution

```bash
# Run only smoke tests
robot --include smoke tests/

# Exclude known broken tests
robot --exclude broken tests/

# Run smoke but not auth
robot --include smoke --exclude auth tests/
```

## Log levels

```robot
*** Test Cases ***
Logging Example
    Log    INFO level message
    Log    DEBUG detail    level=DEBUG
    Log    Warning!    level=WARN
    Log Variables
```

## Best practice: suite skeleton

```robot
*** Settings ***
Documentation    Covers user management workflows.
Library    SeleniumLibrary
Library    Collections
Resource    resources/common.robot
Variables    variables/env.py
Suite Setup    Open App Browser
Suite Teardown    Close App Browser
Test Teardown    Run Keyword If Test Failed    Capture Page Screenshot

*** Variables ***
${ADMIN_USER}    admin

*** Test Cases ***
Create User Account
    [Documentation]    Verifies an admin can create a new user.
    [Tags]    smoke    users
    Log    placeholder
```

## Parallel execution with pabot

```bash
pip install robotframework-pabot
pabot --processes 4 tests/
```

## CI integration

Run from CI with environment-specific config:

```bash
robot \
  --variablefile variables/env.py \
  --variable ENV:staging \
  --include smoke \
  --loglevel INFO \
  --output results/output.xml \
  tests/
```
