import { useMemo } from "react";

interface CurrentUser {
  name: string;
  role: string;
  initials: string;
}

const FALLBACK: CurrentUser = { name: "User", role: "Guest", initials: "?" };

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const part = token.split(".")[1];
    if (!part) return null;
    const json = atob(part.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(json);
  } catch {
    return null;
  }
}

export function useCurrentUser(): CurrentUser {
  return useMemo(() => {
    const raw = localStorage.getItem("access_token");
    if (!raw) return FALLBACK;

    const payload = decodeJwtPayload(raw);
    if (!payload) return FALLBACK;

    const email = typeof payload.email === "string" ? payload.email : "";
    const roles = Array.isArray(payload.roles) ? (payload.roles as string[]) : [];

    const localPart = email.split("@")[0] ?? "";
    const name = localPart
      ? localPart.charAt(0).toUpperCase() + localPart.slice(1)
      : "User";

    const role = roles[0]
      ? roles[0].charAt(0).toUpperCase() + roles[0].slice(1)
      : "Guest";

    const initials = name
      .split(/[._-]/)
      .slice(0, 2)
      .map((w) => w[0]?.toUpperCase() ?? "")
      .join("");

    return { name: name || "User", role, initials: initials || "?" };
  }, []);
}
