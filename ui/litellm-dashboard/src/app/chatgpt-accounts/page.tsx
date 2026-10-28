"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import { $api, fetchClient } from "@/lib/http/api";
import { ApiError } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type View = components["schemas"]["AccountsView"];
type Account = components["schemas"]["AccountRecord"];
type Login = components["schemas"]["LoginView"];

const message = (error: unknown) => (error instanceof Error ? error.message : "Operation failed");
const selectClass = "w-full rounded-md border bg-background p-2 text-sm";

function Assignments({
  view,
  run,
  busy,
}: {
  view: View;
  run: (action: () => Promise<unknown>) => void;
  busy: boolean;
}) {
  const [primary, setPrimary] = useState(view.bindings?.default_account_id ?? "");
  const [automatic, setAutomatic] = useState(view.bindings?.automatic_failover ?? false);
  const [routes, setRoutes] = useState<Record<string, string | null>>(view.bindings?.routes ?? {});
  const options = view.accounts.filter((account) => account.enabled);
  return (
    <section className="space-y-4 rounded-lg border p-5" aria-label="Account assignments">
      <h2 className="text-lg font-semibold">Account assignments</h2>
      <p className="text-sm text-muted-foreground">
        Changes apply to new requests. Existing streams and Responses continuations keep their account. Activation
        checks every affected model.
      </p>
      <form
        className="space-y-4"
        onSubmit={(event) => {
          event.preventDefault();
          run(() =>
            fetchClient.PUT("/chatgpt/accounts/bindings", {
              body: {
                revision: view.revision,
                bindings: { default_account_id: primary || null, routes, automatic_failover: automatic },
              },
            }),
          );
        }}
      >
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={automatic} onChange={(event) => setAutomatic(event.target.checked)} />
          Automatically switch ChatGPT accounts when unavailable
        </label>
        <p className="text-sm text-muted-foreground">
          A failed account is skipped until its retry time or a successful access check. New requests keep using the
          working account. Each account is attempted at most once per request.
        </p>
        {Object.entries(view.active_accounts ?? {}).map(([scope, id]) => {
          const selected = view.accounts.find((account) => account.id === id);
          const change = view.last_switches?.[scope];
          return (
            <div key={scope} className="rounded border p-3 text-sm" role="status">
              <p>
                {scope === "default" ? "Hub current account" : scope}: {selected?.label ?? id} · {id.slice(0, 8)}
              </p>
              {change && (
                <p className="text-muted-foreground">
                  Last switch:{" "}
                  {view.accounts.find((account) => account.id === change.previous)?.label ?? change.previous}
                  {" -> "}
                  {selected?.label ?? id}. Reason: {change.reason}. {new Date(change.at * 1000).toLocaleString()}
                </p>
              )}
            </div>
          );
        })}
        <div>
          <Label htmlFor="primary-account">Hub default</Label>
          <select
            id="primary-account"
            className={selectClass}
            value={primary}
            onChange={(event) => setPrimary(event.target.value)}
            required
          >
            <option value="">Select account</option>
            {options.map((account) => (
              <option key={account.id} value={account.id}>
                {account.label} · {account.email ?? "No email"} · {account.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </div>
        {view.routes
          .filter((route) => ["main-dev", "app-test"].includes(route.name))
          .map((route) => (
            <div key={route.name}>
              <Label htmlFor={`route-${route.name}`}>
                {route.name} · {route.model} · {route.context_length?.toLocaleString() ?? "Provider context"}
              </Label>
              <select
                id={`route-${route.name}`}
                className={selectClass}
                value={routes[route.name] ?? ""}
                onChange={(event) => setRoutes({ ...routes, [route.name]: event.target.value || null })}
              >
                <option value="">Inherit hub default</option>
                {options.map((account) => (
                  <option key={account.id} value={account.id}>
                    {account.label} · {account.email ?? "No email"} · {account.id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </div>
          ))}
        <Button type="submit" disabled={busy || !primary}>
          Verify and apply
        </Button>
      </form>
    </section>
  );
}

function AccountRow({
  account,
  assigned,
  run,
  login,
  busy,
}: {
  account: Account;
  assigned: string[];
  run: (action: () => Promise<unknown>) => void;
  login: (label: string, id?: string) => void;
  busy: boolean;
}) {
  const [label, setLabel] = useState(account.label);
  return (
    <article className="space-y-3 rounded-lg border p-4" aria-label={`Account ${account.label}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold">{account.label}</h3>
          <p className="text-sm">{account.email ?? "Email unavailable"}</p>
          <p className="text-xs text-muted-foreground break-all">Profile {account.id}</p>
        </div>
        <strong className="text-sm">{account.status}</strong>
      </div>
      <p className="text-sm">Assigned: {assigned.join(", ") || "None"}</p>
      {Object.entries(account.deployment_outages ?? {}).map(([model, outage]) => (
        <p key={model} className="text-sm text-muted-foreground">
          {model}: {outage.reason}. {outage.retry_at ? "Provider retry" : "Local retry"}{" "}
          {outage.until ? new Date(outage.until * 1000).toLocaleString() : "Unavailable"}
        </p>
      ))}
      <p className="text-sm text-muted-foreground">
        Last check: {account.last_checked_at ? new Date(account.last_checked_at * 1000).toLocaleString() : "Never"}.
        Provider retry time: {account.retry_at ? new Date(account.retry_at * 1000).toLocaleString() : "Unavailable"}.
        Local retry:{" "}
        {account.unavailable_until ? new Date(account.unavailable_until * 1000).toLocaleString() : "Not scheduled"}
      </p>
      {account.error_code && <p role="status">{account.error_code}</p>}
      <p className="text-sm text-muted-foreground">Quota, reset, subscription expiry and credits: unavailable</p>
      <details>
        <summary className="cursor-pointer text-sm">Available models and OAuth client</summary>
        <p className="text-xs break-all">{account.models?.join(", ") || "Not checked"}</p>
        <p className="text-xs break-all">OAuth client: {account.client_id}</p>
        <p className="text-xs break-all">Workspace: {account.workspace ?? "Unavailable"}</p>
      </details>
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          run(() =>
            fetchClient.PATCH("/chatgpt/accounts/{account_id}", {
              params: { path: { account_id: account.id } },
              body: { label },
            }),
          );
        }}
      >
        <Label className="sr-only" htmlFor={`label-${account.id}`}>
          Label for {account.label}
        </Label>
        <Input
          id={`label-${account.id}`}
          value={label}
          maxLength={120}
          required
          onChange={(event) => setLabel(event.target.value)}
        />
        <Button variant="outline" type="submit" disabled={busy}>
          Rename
        </Button>
      </form>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          disabled={busy || !account.enabled}
          onClick={() =>
            run(() =>
              fetchClient.POST("/chatgpt/accounts/{account_id}/check", {
                params: { path: { account_id: account.id } },
              }),
            )
          }
        >
          Check access
        </Button>
        <Button variant="outline" disabled={busy} onClick={() => login(account.label, account.id)}>
          Sign in again
        </Button>
        <Button
          variant="outline"
          disabled={busy || !account.enabled || assigned.length > 0}
          onClick={() =>
            run(() =>
              fetchClient.POST("/chatgpt/accounts/{account_id}/disconnect", {
                params: { path: { account_id: account.id } },
              }),
            )
          }
        >
          Disconnect
        </Button>
      </div>
      {assigned.length > 0 && (
        <p className="text-xs text-muted-foreground">Reassign this account before disconnecting.</p>
      )}
    </article>
  );
}

function DeviceLogin({
  operation,
  run,
  done,
}: {
  operation: Login;
  run: (action: () => Promise<unknown>) => void;
  done: () => void;
}) {
  const query = $api.useQuery(
    "get",
    "/chatgpt/accounts/logins/{login_id}",
    { params: { path: { login_id: operation.id } } },
    { retry: false, refetchInterval: (state) => (state.state.data?.status === "pending" ? 2000 : false) },
  );
  const current = query.data ?? operation;
  return (
    <section className="space-y-3 rounded-lg border border-blue-400 p-5" aria-label="ChatGPT sign-in">
      <h2 className="font-semibold">ChatGPT sign-in: {current.status}</h2>
      {current.status === "pending" ? (
        <>
          <p>
            Open{" "}
            <a className="underline" href={current.verification_url} target="_blank" rel="noopener noreferrer">
              ChatGPT device authorization
            </a>{" "}
            and enter:
          </p>
          <p className="font-mono text-2xl">{current.user_code}</p>
          <p>Expires {new Date(current.expires_at * 1000).toLocaleTimeString()}. Your current account stays active.</p>
          <Button
            variant="outline"
            onClick={() =>
              run(() =>
                fetchClient.DELETE("/chatgpt/accounts/logins/{login_id}", {
                  params: { path: { login_id: operation.id } },
                }),
              )
            }
          >
            Cancel sign-in
          </Button>
        </>
      ) : (
        <>
          <p role="status">
            {current.status === "ready"
              ? "Account verified. Select it in account assignments."
              : current.error_code ?? "Sign-in was not completed."}
          </p>
          <Button onClick={done}>Close and refresh accounts</Button>
        </>
      )}
      {query.error && <p role="alert">{message(query.error)}</p>}
    </section>
  );
}

export default function ChatGPTAccountsPage() {
  const client = useQueryClient();
  const [key, setKey] = useState("");
  const [label, setLabel] = useState("");
  const [operation, setOperation] = useState<Login | null>(null);
  const accounts = $api.useQuery("get", "/chatgpt/accounts", {}, { retry: false, refetchInterval: 30000 });
  const mutation = useMutation({
    mutationFn: (action: () => Promise<unknown>) => action(),
    onSuccess: () => client.invalidateQueries({ queryKey: ["get", "/chatgpt/accounts"] }),
  });
  const run = (action: () => Promise<unknown>) => mutation.mutate(action);
  const login = (name: string, id?: string) =>
    run(async () => {
      const result = await fetchClient.POST("/chatgpt/accounts/logins", {
        body: { label: name, account_id: id ?? null },
      });
      if (result.data) setOperation(result.data);
    });
  const accountsError: unknown = accounts.error;
  const unauthorized = accountsError instanceof ApiError && accountsError.status === 401;
  return (
    <main className="mx-auto max-w-5xl space-y-6 p-6 md:p-10">
      <header>
        <Link href="/ui/" className="text-sm underline">
          LiteLLM Dashboard
        </Link>
        <h1 className="mt-4 text-3xl font-semibold">ChatGPT accounts</h1>
        <p className="mt-2 text-muted-foreground">
          Connect separate accounts, choose a preferred account, and enable automatic failover when needed.
        </p>
      </header>
      {mutation.error && (
        <p role="alert" className="rounded border border-red-400 p-3">
          {message(mutation.error)}
        </p>
      )}
      {unauthorized && (
        <form
          className="max-w-md space-y-3 rounded-lg border p-5"
          onSubmit={(event) => {
            event.preventDefault();
            run(async () => {
              await fetchClient.POST("/chatgpt/accounts/session", { body: { key } });
              setKey("");
            });
          }}
        >
          <h2 className="font-semibold">Administrator sign-in</h2>
          <Label htmlFor="admin-key">Account management key</Label>
          <Input
            id="admin-key"
            type="password"
            autoComplete="current-password"
            value={key}
            required
            onChange={(event) => setKey(event.target.value)}
          />
          <p className="text-sm text-muted-foreground">
            The inference key cannot manage accounts. This session lasts one hour.
          </p>
          <Button type="submit" disabled={mutation.isPending}>
            Sign in
          </Button>
        </form>
      )}
      {accounts.isPending && <p role="status">Loading accounts…</p>}
      {Boolean(accountsError) && !unauthorized && <p role="alert">{message(accounts.error)}</p>}
      {accounts.data && !unauthorized && (
        <>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => run(() => fetchClient.DELETE("/chatgpt/accounts/session"))}>
              Sign out
            </Button>
            <Button variant="outline" onClick={() => void accounts.refetch()}>
              Refresh
            </Button>
          </div>
          <form
            className="flex flex-wrap items-end gap-3 rounded-lg border p-5"
            onSubmit={(event) => {
              event.preventDefault();
              login(label);
            }}
          >
            <div className="grow space-y-2">
              <Label htmlFor="new-account-label">New account label</Label>
              <Input
                id="new-account-label"
                placeholder="Personal or Work"
                maxLength={120}
                value={label}
                required
                onChange={(event) => setLabel(event.target.value)}
              />
            </div>
            <Button type="submit" disabled={mutation.isPending || operation?.status === "pending"}>
              Add ChatGPT account
            </Button>
          </form>
          {operation && (
            <DeviceLogin
              operation={operation}
              run={run}
              done={() => {
                setOperation(null);
                void accounts.refetch();
              }}
            />
          )}
          <Assignments key={accounts.data.revision} view={accounts.data} run={run} busy={mutation.isPending} />
          <section className="grid gap-4 md:grid-cols-2" aria-label="Connected accounts">
            {accounts.data.accounts.length === 0 && <p>No ChatGPT accounts connected yet.</p>}
            {accounts.data.accounts.map((account) => (
              <AccountRow
                key={`${account.id}-${account.label}`}
                account={account}
                assigned={[
                  ...(accounts.data.bindings?.default_account_id === account.id ? ["Hub default"] : []),
                  ...Object.entries(accounts.data.active_accounts ?? {})
                    .filter(([, id]) => id === account.id)
                    .map(([scope]) => `Active ${scope}`),
                  ...accounts.data.routes
                    .filter(
                      (route) =>
                        (accounts.data.bindings?.routes?.[route.name] ?? accounts.data.bindings?.default_account_id) ===
                        account.id,
                    )
                    .map((route) => route.name),
                ]}
                run={run}
                login={login}
                busy={mutation.isPending}
              />
            ))}
          </section>
        </>
      )}
    </main>
  );
}
