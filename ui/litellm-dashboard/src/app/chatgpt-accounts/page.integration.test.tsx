import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import Page from "./page";

const first = "00000000-0000-4000-8000-000000000001";
const second = "00000000-0000-4000-8000-000000000002";
const account = (id: string) => ({
  id,
  label: "Same label",
  email: "same@example.test",
  client_id: "fixture",
  enabled: true,
  status: "ready",
  models: [],
});

afterEach(() => vi.unstubAllGlobals());

it("uses independent account IDs for identical labels and submits route overrides", async () => {
  const requests: { path: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: Request) => {
      const path = new URL(request.url).pathname;
      if (request.method === "GET") {
        const view = {
          revision: 7,
          version: 1,
          accounts: [account(first), account(second)],
          bindings: { default_account_id: first, routes: {} },
          routes: [
            { name: "main-dev", model: "chatgpt/gpt-6.1-terra", context_length: 500000 },
            { name: "app-test", model: "chatgpt/gpt-6-luna", context_length: 272000 },
          ],
        };
        return Response.json(view);
      }
      requests.push({ path, body: await request.json() });
      return Response.json({});
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <Page />
    </QueryClientProvider>,
  );
  const primary = await screen.findByLabelText("Hub default");
  fireEvent.change(primary, { target: { value: second } });
  fireEvent.change(screen.getByLabelText(/app-test ·/), { target: { value: first } });
  fireEvent.click(screen.getByRole("checkbox", { name: "Automatically switch ChatGPT accounts when unavailable" }));
  fireEvent.click(screen.getByRole("button", { name: "Verify and apply" }));
  await screen.findByRole("button", { name: "Verify and apply" });
  await vi.waitFor(() =>
    expect(requests).toEqual([
      {
        path: "/chatgpt/accounts/bindings",
        body: {
          revision: 7,
          bindings: { default_account_id: second, routes: { "app-test": first }, automatic_failover: true },
        },
      },
    ]),
  );
  expect(screen.getAllByRole("button", { name: "Disconnect" })[0]).toBeDisabled();
  client.clear();
});

it("hides cached account details after the administrative session ends", async () => {
  let authenticated = true;
  const view = {
    revision: 1,
    version: 1,
    accounts: [account(first)],
    bindings: { default_account_id: first, routes: {} },
    routes: [{ name: "main-dev", model: "chatgpt/gpt-6.1-terra", context_length: 500000 }],
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: Request) => {
      if (request.method === "DELETE") {
        authenticated = false;
        return new Response(null, { status: 204 });
      }
      return authenticated
        ? Response.json(view)
        : Response.json({ detail: "Administrative session required" }, { status: 401 });
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <Page />
    </QueryClientProvider>,
  );
  expect(await screen.findByLabelText("Hub default")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Sign out" }));
  expect(await screen.findByRole("heading", { name: "Administrator sign-in" })).toBeInTheDocument();
  expect(screen.queryByLabelText("Hub default")).not.toBeInTheDocument();
  expect(screen.queryByRole("article", { name: "Account Same label" })).not.toBeInTheDocument();
  client.clear();
});
