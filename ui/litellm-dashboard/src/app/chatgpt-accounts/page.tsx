"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import { Trans, useTranslation } from "react-i18next";
import { $api, fetchClient } from "@/lib/http/api";
import { ApiError } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type View = components["schemas"]["AccountsView"];
type Account = components["schemas"]["AccountRecord"];
type Login = components["schemas"]["LoginView"];

const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);
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
  const { t } = useTranslation();
  const [primary, setPrimary] = useState(view.bindings?.default_account_id ?? "");
  const [automatic, setAutomatic] = useState(view.bindings?.automatic_failover ?? false);
  const [routes, setRoutes] = useState<Record<string, string | null>>(view.bindings?.routes ?? {});
  const options = view.accounts.filter((account) => account.enabled);
  return (
    <section className="space-y-4 rounded-lg border p-5" aria-label={t("chatgptAccounts.assignmentsTitle")}>
      <h2 className="text-lg font-semibold">{t("chatgptAccounts.assignmentsTitle")}</h2>
      <p className="text-sm text-muted-foreground">{t("chatgptAccounts.assignmentsHelp")}</p>
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
          {t("chatgptAccounts.automaticFailover")}
        </label>
        <p className="text-sm text-muted-foreground">{t("chatgptAccounts.automaticHelp")}</p>
        {Object.entries(view.active_accounts ?? {}).map(([scope, id]) => {
          const selected = view.accounts.find((account) => account.id === id);
          const change = view.last_switches?.[scope];
          const switchDetails = change && {
            previous: view.accounts.find((account) => account.id === change.previous)?.label ?? change.previous,
            current: selected?.label ?? id,
            reason: change.reason,
            time: new Date(change.at * 1000).toLocaleString(),
          };
          return (
            <div key={scope} className="rounded border p-3 text-sm" role="status">
              <p>
                {scope === "default" ? t("chatgptAccounts.currentAccount") : scope}: {selected?.label ?? id} ·{" "}
                {id.slice(0, 8)}
              </p>
              {switchDetails && (
                <p className="text-muted-foreground">{t("chatgptAccounts.lastSwitch", switchDetails)}</p>
              )}
            </div>
          );
        })}
        <div>
          <Label htmlFor="primary-account">{t("chatgptAccounts.hubDefault")}</Label>
          <select
            id="primary-account"
            className={selectClass}
            value={primary}
            onChange={(event) => setPrimary(event.target.value)}
            required
          >
            <option value="">{t("chatgptAccounts.selectAccount")}</option>
            {options.map((account) => (
              <option key={account.id} value={account.id}>
                {account.label} · {account.email ?? t("chatgptAccounts.noEmail")} · {account.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </div>
        {view.routes
          .filter((route) => ["main-dev", "app-test"].includes(route.name))
          .map((route) => (
            <div key={route.name}>
              <Label htmlFor={`route-${route.name}`}>
                {route.name} · {route.model} ·{" "}
                {route.context_length?.toLocaleString() ?? t("chatgptAccounts.providerContext")}
              </Label>
              <select
                id={`route-${route.name}`}
                className={selectClass}
                value={routes[route.name] ?? ""}
                onChange={(event) => setRoutes({ ...routes, [route.name]: event.target.value || null })}
              >
                <option value="">{t("chatgptAccounts.inheritDefault")}</option>
                {options.map((account) => (
                  <option key={account.id} value={account.id}>
                    {account.label} · {account.email ?? t("chatgptAccounts.noEmail")} · {account.id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </div>
          ))}
        <Button type="submit" disabled={busy || !primary}>
          {t("chatgptAccounts.verifyApply")}
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
  const { t } = useTranslation();
  const [label, setLabel] = useState(account.label);
  return (
    <article
      className="space-y-3 rounded-lg border p-4"
      aria-label={t("chatgptAccounts.accountLabel", { label: account.label })}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold">{account.label}</h3>
          <p className="text-sm">{account.email ?? t("chatgptAccounts.emailUnavailable")}</p>
          <p className="text-xs text-muted-foreground break-all">{t("chatgptAccounts.profile", { id: account.id })}</p>
        </div>
        <strong className="text-sm">{account.status}</strong>
      </div>
      <p className="text-sm">
        {t("chatgptAccounts.assigned", { assignments: assigned.join(", ") || t("chatgptAccounts.none") })}
      </p>
      {Object.entries(account.deployment_outages ?? {}).map(([model, outage]) => (
        <p key={model} className="text-sm text-muted-foreground">
          {model}: {outage.reason}.{" "}
          {outage.retry_at ? t("chatgptAccounts.providerRetry") : t("chatgptAccounts.localRetry")}{" "}
          {outage.until ? new Date(outage.until * 1000).toLocaleString() : t("chatgptAccounts.unavailable")}
        </p>
      ))}
      <p className="text-sm text-muted-foreground">
        {t("chatgptAccounts.checkHistory", {
          lastCheck: account.last_checked_at
            ? new Date(account.last_checked_at * 1000).toLocaleString()
            : t("chatgptAccounts.never"),
          providerRetry: account.retry_at
            ? new Date(account.retry_at * 1000).toLocaleString()
            : t("chatgptAccounts.unavailable"),
          localRetry: account.unavailable_until
            ? new Date(account.unavailable_until * 1000).toLocaleString()
            : t("chatgptAccounts.notScheduled"),
        })}
      </p>
      {account.error_code && <p role="status">{account.error_code}</p>}
      <p className="text-sm text-muted-foreground">{t("chatgptAccounts.limitsUnavailable")}</p>
      <details>
        <summary className="cursor-pointer text-sm">{t("chatgptAccounts.modelDetails")}</summary>
        <p className="text-xs break-all">{account.models?.join(", ") || t("chatgptAccounts.notChecked")}</p>
        <p className="text-xs break-all">{t("chatgptAccounts.oauthClient", { clientId: account.client_id })}</p>
        <p className="text-xs break-all">
          {t("chatgptAccounts.workspace", { workspace: account.workspace ?? t("chatgptAccounts.unavailable") })}
        </p>
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
          {t("chatgptAccounts.labelFor", { label: account.label })}
        </Label>
        <Input
          id={`label-${account.id}`}
          value={label}
          maxLength={120}
          required
          onChange={(event) => setLabel(event.target.value)}
        />
        <Button variant="outline" type="submit" disabled={busy}>
          {t("chatgptAccounts.rename")}
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
          {t("chatgptAccounts.checkAccess")}
        </Button>
        <Button variant="outline" disabled={busy} onClick={() => login(account.label, account.id)}>
          {t("chatgptAccounts.signInAgain")}
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
          {t("chatgptAccounts.disconnect")}
        </Button>
      </div>
      {assigned.length > 0 && <p className="text-xs text-muted-foreground">{t("chatgptAccounts.reassignRequired")}</p>}
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
  const { t } = useTranslation();
  const query = $api.useQuery(
    "get",
    "/chatgpt/accounts/logins/{login_id}",
    { params: { path: { login_id: operation.id } } },
    { retry: false, refetchInterval: (state) => (state.state.data?.status === "pending" ? 2000 : false) },
  );
  const current = query.data ?? operation;
  return (
    <section className="space-y-3 rounded-lg border border-blue-400 p-5" aria-label={t("chatgptAccounts.signInLabel")}>
      <h2 className="font-semibold">{t("chatgptAccounts.signInStatus", { status: current.status })}</h2>
      {current.status === "pending" ? (
        <>
          <p>
            <Trans
              i18nKey="chatgptAccounts.deviceInstructions"
              components={{
                authorize: (
                  <a className="underline" href={current.verification_url} target="_blank" rel="noopener noreferrer" />
                ),
              }}
            />
          </p>
          <p className="font-mono text-2xl">{current.user_code}</p>
          <p>{t("chatgptAccounts.expiresHelp", { time: new Date(current.expires_at * 1000).toLocaleTimeString() })}</p>
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
            {t("chatgptAccounts.cancelSignIn")}
          </Button>
        </>
      ) : (
        <>
          <p role="status">
            {current.status === "ready"
              ? t("chatgptAccounts.verified")
              : current.error_code ?? t("chatgptAccounts.loginIncomplete")}
          </p>
          <Button onClick={done}>{t("chatgptAccounts.closeRefresh")}</Button>
        </>
      )}
      {query.error && <p role="alert">{message(query.error, t("chatgptAccounts.operationFailed"))}</p>}
    </section>
  );
}

export default function ChatGPTAccountsPage() {
  const { t } = useTranslation();
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
          {t("chatgptAccounts.dashboard")}
        </Link>
        <h1 className="mt-4 text-3xl font-semibold">{t("shell.nav.items.chatgptAccounts")}</h1>
        <p className="mt-2 text-muted-foreground">{t("chatgptAccounts.pageHelp")}</p>
      </header>
      {mutation.error && (
        <p role="alert" className="rounded border border-red-400 p-3">
          {message(mutation.error, t("chatgptAccounts.operationFailed"))}
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
          <h2 className="font-semibold">{t("chatgptAccounts.adminSignIn")}</h2>
          <Label htmlFor="admin-key">{t("chatgptAccounts.adminKey")}</Label>
          <Input
            id="admin-key"
            type="password"
            autoComplete="current-password"
            value={key}
            required
            onChange={(event) => setKey(event.target.value)}
          />
          <p className="text-sm text-muted-foreground">{t("chatgptAccounts.adminSessionHelp")}</p>
          <Button type="submit" disabled={mutation.isPending}>
            {t("chatgptAccounts.signIn")}
          </Button>
        </form>
      )}
      {accounts.isPending && <p role="status">{t("chatgptAccounts.loading")}</p>}
      {Boolean(accountsError) && !unauthorized && (
        <p role="alert">{message(accounts.error, t("chatgptAccounts.operationFailed"))}</p>
      )}
      {accounts.data && !unauthorized && (
        <>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => run(() => fetchClient.DELETE("/chatgpt/accounts/session"))}>
              {t("chatgptAccounts.signOut")}
            </Button>
            <Button variant="outline" onClick={() => void accounts.refetch()}>
              {t("chatgptAccounts.refresh")}
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
              <Label htmlFor="new-account-label">{t("chatgptAccounts.newLabel")}</Label>
              <Input
                id="new-account-label"
                placeholder={t("chatgptAccounts.labelPlaceholder")}
                maxLength={120}
                value={label}
                required
                onChange={(event) => setLabel(event.target.value)}
              />
            </div>
            <Button type="submit" disabled={mutation.isPending || operation?.status === "pending"}>
              {t("chatgptAccounts.addAccount")}
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
          <section className="grid gap-4 md:grid-cols-2" aria-label={t("chatgptAccounts.connectedLabel")}>
            {accounts.data.accounts.length === 0 && <p>{t("chatgptAccounts.noAccounts")}</p>}
            {accounts.data.accounts.map((account) => (
              <AccountRow
                key={`${account.id}-${account.label}`}
                account={account}
                assigned={[
                  ...(accounts.data.bindings?.default_account_id === account.id
                    ? [t("chatgptAccounts.hubDefault")]
                    : []),
                  ...Object.entries(accounts.data.active_accounts ?? {})
                    .filter(([, id]) => id === account.id)
                    .map(([scope]) => t("chatgptAccounts.activeScope", { scope })),
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
