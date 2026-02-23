# Robot Framework Standard Libraries (Seeded Authoritative Summary)

Primary references:
- https://robotframework.org/robotframework/latest/libraries/BuiltIn.html
- https://robotframework.org/robotframework/latest/libraries/Collections.html
- https://robotframework.org/robotframework/latest/libraries/OperatingSystem.html
- https://robotframework.org/robotframework/latest/libraries/String.html

## BuiltIn library

BuiltIn is available automatically and provides core keywords for:
- flow control helpers
- variable handling (`Set Variable`, conversions, checks)
- assertions (`Should Be Equal`, `Should Contain`, etc.)
- logging (`Log`, `Log To Console`)
- running keywords conditionally or repeatedly (`Run Keyword If`, retry-style patterns)

## Collections library

Collections helps with structured test data:
- list and dictionary manipulation
- membership/length/content assertions
- item retrieval and updates

Use it to keep data handling explicit instead of embedding complex Python expressions in test data.

## OperatingSystem library

OperatingSystem supports filesystem and process-oriented tasks:
- file and directory checks
- path operations
- environment variable access
- process execution helpers

For CI reliability:
- use explicit paths
- check existence before operations
- avoid platform assumptions in keywords

## String library

String provides common text transforms and validations:
- case normalization
- splitting/joining
- substring checks
- replace/strip operations

Consistent normalization helps reduce flaky assertions around whitespace/casing differences.

## Best-practice usage pattern

For enterprise-grade suites:
- keep business intent in user keywords
- use standard libraries for low-level operations
- centralize cross-cutting behavior in resource keywords (logging, retries, cleanup)

