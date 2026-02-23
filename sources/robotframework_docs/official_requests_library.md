# RequestsLibrary for Robot Framework (Authoritative Summary)

Primary references:
- https://marketsquare.github.io/robotframework-requests/doc/RequestsLibrary.html
- https://github.com/MarketSquare/robotframework-requests

## Overview

RequestsLibrary wraps the Python `requests` library for HTTP API testing in Robot Framework.
Install via: `pip install robotframework-requests`

## Core import

```robot
*** Settings ***
Library    RequestsLibrary
```

## Session management

Sessions must be created before sending requests. Reuse them across tests via Suite Setup:

```robot
*** Settings ***
Library    RequestsLibrary
Suite Setup    Create Session    api    https://api.example.com
Suite Teardown    Delete All Sessions
```

## Core keywords

- `Create Session    alias    url    headers=&{headers}    timeout=30` — creates named session
- `GET On Session    alias    endpoint` — HTTP GET, returns response object
- `POST On Session    alias    endpoint    json=&{body}` — HTTP POST with JSON body
- `PUT On Session    alias    endpoint    json=&{body}` — HTTP PUT
- `DELETE On Session    alias    endpoint`
- `Delete All Sessions`

## Simple GET request

```robot
*** Settings ***
Library    RequestsLibrary

*** Test Cases ***
Verify API Health
    Create Session    api    https://jsonplaceholder.typicode.com
    ${resp}=    GET On Session    api    /todos/1
    Should Be Equal As Integers    ${resp.status_code}    200
```

## POST with JSON body

```robot
*** Settings ***
Library    RequestsLibrary
Library    Collections

*** Test Cases ***
Create New Post
    ${body}=    Create Dictionary    title=foo    body=bar    userId=1
    Create Session    api    https://jsonplaceholder.typicode.com
    ${resp}=    POST On Session    api    /posts    json=${body}
    Should Be Equal As Integers    ${resp.status_code}    201
```

## Parse JSON response field

```robot
*** Settings ***
Library    RequestsLibrary

*** Test Cases ***
Assert Response Field
    Create Session    api    https://jsonplaceholder.typicode.com
    ${resp}=    GET On Session    api    /todos/1
    ${body}=    Set Variable    ${resp.json()}
    Should Be Equal As Integers    ${body}[id]    1
```

## Bearer token authentication

```robot
*** Settings ***
Library    RequestsLibrary
Library    Collections

*** Keywords ***
Create Auth Session
    [Arguments]    ${token}
    ${headers}=    Create Dictionary    Authorization=Bearer ${token}
    Create Session    api    https://api.example.com    headers=${headers}
```

## Reuse session across tests

```robot
*** Settings ***
Library    RequestsLibrary
Suite Setup    Create Shared Session
Suite Teardown    Delete All Sessions

*** Test Cases ***
First Endpoint
    ${r}=    GET On Session    shared    /todos/1
    Should Be Equal As Integers    ${r.status_code}    200

Second Endpoint
    ${r}=    GET On Session    shared    /todos/2
    Should Be Equal As Integers    ${r.status_code}    200

*** Keywords ***
Create Shared Session
    Create Session    shared    https://jsonplaceholder.typicode.com
```

## DELETE request

```robot
*** Settings ***
Library    RequestsLibrary

*** Test Cases ***
Delete A Resource
    Create Session    api    https://jsonplaceholder.typicode.com
    ${resp}=    DELETE On Session    api    /posts/1
    Should Be True    ${resp.status_code} in [200, 204]
```

## Timeout and error handling

Set `timeout` on `Create Session` or per-request:

```robot
*** Settings ***
Library    RequestsLibrary

*** Test Cases ***
Request With Timeout
    Create Session    api    https://httpbin.org    timeout=5
    TRY
        ${resp}=    GET On Session    api    /delay/10
    EXCEPT    *Timeout*    type=GLOB
        Log    Request timed out as expected
    END
```

## Response assertions

```robot
*** Test Cases ***
Assert JSON Field
    Create Session    api    https://jsonplaceholder.typicode.com
    ${resp}=    GET On Session    api    /posts/1
    Should Be Equal As Integers    ${resp.status_code}    200
    ${json}=    Set Variable    ${resp.json()}
    Should Be Equal    ${json}[title]    sunt aut facere repellat provident
```
