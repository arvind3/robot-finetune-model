# SeleniumLibrary for Robot Framework (Authoritative Summary)

Primary references:
- https://robotframework.org/SeleniumLibrary/SeleniumLibrary.html
- https://github.com/robotframework/SeleniumLibrary

## Overview

SeleniumLibrary is the primary web browser automation library for Robot Framework.
Install via: `pip install robotframework-seleniumlibrary`

## Core import

```robot
*** Settings ***
Library    SeleniumLibrary
```

## Essential keywords

- `Open Browser    url    browser` — opens a browser (chrome, firefox, headlessChrome)
- `Close Browser` / `Close All Browsers`
- `Go To    url`
- `Input Text    locator    text`
- `Input Password    locator    password`
- `Click Element    locator` / `Click Button    locator`
- `Wait Until Element Is Visible    locator    timeout=10s`
- `Wait Until Page Contains    text    timeout=15s`
- `Element Should Be Visible    locator`
- `Page Should Contain    text`
- `Title Should Be    title`
- `Capture Page Screenshot`
- `Get Text    locator` — returns element text
- `Element Should Contain    locator    text`
- `Select From List By Label    locator    label`
- `Get Selected List Label    locator`

## Open browser and verify title

```robot
*** Settings ***
Library    SeleniumLibrary
Suite Teardown    Close All Browsers

*** Test Cases ***
Verify Homepage Title
    Open Browser    https://example.com    chrome
    Title Should Be    Example Domain
```

## Fill a form and submit

```robot
*** Settings ***
Library    SeleniumLibrary

*** Test Cases ***
Submit Login Form
    Open Browser    https://example.com/login    chrome
    Input Text    id=username    testuser
    Input Password    id=password    secret
    Click Button    id=submit
    Wait Until Page Contains    Welcome
    [Teardown]    Close Browser
```

## Wait before interacting

```robot
*** Settings ***
Library    SeleniumLibrary

*** Test Cases ***
Click After Wait
    Open Browser    https://example.com    chrome
    Wait Until Element Is Visible    id=loadedBtn    timeout=10s
    Click Element    id=loadedBtn
    [Teardown]    Close Browser
```

## Screenshot on failure

```robot
*** Settings ***
Library    SeleniumLibrary
Test Teardown    Run Keyword If Test Failed    Capture Page Screenshot

*** Test Cases ***
Login Test
    Open Browser    https://example.com/login    chrome
    Input Text    id=user    admin
    Click Button    id=go
```

## Locator strategies

Supported: `id=`, `name=`, `css=`, `xpath=`, `class=`, `link=`, `partial link=`.
Prefer `id=` and `css=` for maintainability. Avoid absolute XPath.

## Reliability pattern — safe click

```robot
*** Keywords ***
Safe Click Element
    [Arguments]    ${locator}
    Wait Until Element Is Visible    ${locator}    timeout=15s
    Click Element    ${locator}
```

## Page Object resource pattern

Resource file `resources/login_page.robot`:
```robot
*** Settings ***
Library    SeleniumLibrary

*** Keywords ***
Open Login Page
    Go To    https://example.com/login

Enter Credentials
    [Arguments]    ${user}    ${pass}
    Input Text    id=username    ${user}
    Input Password    id=password    ${pass}

Submit Login
    Click Button    id=submit
```

Main suite:
```robot
*** Settings ***
Library    SeleniumLibrary
Resource    resources/login_page.robot
Suite Setup    Open Browser    about:blank    chrome
Suite Teardown    Close All Browsers

*** Test Cases ***
Login With Valid Creds
    Open Login Page
    Enter Credentials    admin    s3cret
    Submit Login
    Wait Until Page Contains    Dashboard
```

## Dropdown handling

```robot
*** Settings ***
Library    SeleniumLibrary

*** Test Cases ***
Select Country Dropdown
    Open Browser    https://example.com/form    chrome
    Select From List By Label    id=country    Germany
    ${val}=    Get Selected List Label    id=country
    Should Be Equal    ${val}    Germany
    [Teardown]    Close Browser
```

## Headless execution

Pass `headlessChrome` or `headlessFirefox` as the browser argument for CI environments:
```robot
*** Test Cases ***
Headless Test
    Open Browser    https://example.com    headlessChrome
    Title Should Be    Example Domain
    [Teardown]    Close Browser
```
