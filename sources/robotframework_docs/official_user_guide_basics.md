# Robot Framework User Guide Basics (Seeded Authoritative Summary)

Primary references:
- https://robotframework.org/robotframework/index.html
- https://docs.robotframework.org/
- https://docs.robotframework.org/docs

## Core data model

Robot Framework test data is keyword-driven and organized using tables. The most common sections are:
- `*** Settings ***`
- `*** Variables ***`
- `*** Test Cases ***` (or `*** Tasks ***` in RPA usage)
- `*** Keywords ***`

Test suites can be split into multiple files and folders. Shared behavior is typically moved into:
- resource files (`Resource`)
- variable files (`Variables`)
- libraries (`Library`)

## Execution and suite structure

Recommended structure for maintainability:
- suite-level setup/teardown for shared environment lifecycle
- small, intention-revealing test cases
- reusable keywords with clear names and arguments
- avoid large monolithic test files

Useful execution controls:
- include/exclude tags
- variable overrides at runtime
- dry run for early syntax verification

## Keyword design expectations

Good keyword design favors:
- one clear responsibility per keyword
- explicit arguments and return values
- minimal hidden side effects
- log output that helps failure diagnosis

## Reliability guidance

For stable automation:
- prefer condition-based waits and retries over fixed sleeps
- isolate unstable UI/API operations behind robust wrapper keywords
- capture context in teardown on failure

## Data-driven style

Robot Framework supports data-driven testing with templates, loops, and variables. Keep data and behavior separated where possible so tests stay readable as scenarios grow.

## Diagnostic practices

When failures happen, first verify:
- imported libraries/resources are correct
- keyword names and arguments match definitions
- environment assumptions (paths, credentials, services) are valid

Then reduce scope to a minimal reproducer and iterate from a passing baseline.

