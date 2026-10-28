# ChatGPT accounts

Manage ChatGPT accounts in a single-worker, file-backed proxy without PostgreSQL or Redis

```bash
export LITELLM_CHATGPT_ACCOUNTS_DIR=/var/lib/litellm/chatgpt-accounts
export LITELLM_CHATGPT_ACCOUNTS_ADMIN_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

Keep the management key separate from the inference key, in a protected environment file owned by the operator
The state directory must belong to the runtime user and remain private
Start one proxy worker with the usual model list, including ChatGPT deployments with `model_info.mode: responses`
Build the dashboard using its locked Node and npm versions and `npm ci && npm run gen:api && npm run build`
Serve the resulting `out` through the existing `LITELLM_UI_PATH` option

Open `/ui/chatgpt-accounts/` and sign in with the management key
The administrator session uses an httpOnly SameSite Strict cookie and expires after one hour
An inference bearer key cannot list or modify profiles
Use HTTPS when accessing the administrative page outside a trusted local network

Add an account using the displayed device link and code
The operation has a deadline and can be cancelled without changing an existing account
The account becomes ready only after a model access probe succeeds
Each account has an immutable UUID and its own `credentials/<UUID>/auth.json`; labels and email addresses may repeat

Select the hub default, and optionally override `main-dev` or `app-test`
Direct ChatGPT models inherit the hub default
Activation probes every affected model before committing the new bindings
New requests use the new selection and Responses continuations retain their original account
A continuation whose original account is missing or disabled fails instead of using another account
Reassign an account before disconnecting it; disconnecting does not terminate an existing stream

The manager creates stable Router deployments for each profile and model
It never changes environment variables per request
Automatic account failover is opt-in through the assignments form or `bindings.automatic_failover`
When enabled, a failed account is skipped and the working account remains selected until it fails
Each request attempts an account at most once and never uses another provider as a hidden fallback
Quota and billing failures affect the account, while model access failures and transient errors affect its deployment
An observed Retry-After controls recovery, with a local 600-second retry for unknown quota resets and 30 seconds for transient errors
A failed reserve does not prevent manually assigning a working account
Invalid requests do not rotate, and a partial stream is never replayed on another account
The complete inference request has a 600-second deadline, including retries and streaming
`LITELLM_CHATGPT_ACCOUNTS_REQUEST_TIMEOUT` can shorten that deadline but cannot increase it
When every account is unavailable, the proxy returns a clear `503` with Retry-After only when a retry time is known
With automatic failover disabled, the existing provider fallback configuration remains in effect

Token refresh is locked per file, uses atomic private writes and never starts an interactive login from inference
A ChatGPT `401 token_invalidated` refreshes the selected profile and retries the same request once
Other authentication failures, `403`, `429` and other providers do not trigger that retry
Explicit command-line device login is now `Authenticator().login()`; `get_access_token()` does not start a login

The API under `/chatgpt/accounts` supports sessions, pending logins, account checks, renaming, disconnecting and bindings
Credentials are never included in its response models
Client parameters cannot override managed credential files or account headers
Account status, deployment failures, active selections and switch reasons come from observed provider results
Unavailable quota and credit information remains unavailable
Access-token expiry is not a subscription expiry or quota reset

For rollback, restore the previous runtime and configuration while retaining the newest credentials
Do not restore stale refresh-token backups automatically
The pilot requires separate test accounts; do not copy a live production refresh token into a test installation
