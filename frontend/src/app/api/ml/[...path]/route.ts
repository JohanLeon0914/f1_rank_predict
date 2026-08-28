import { NextRequest, NextResponse } from "next/server";

const API_BASE_URL = process.env.ML_API_BASE_URL ?? "http://localhost:8000";

async function proxy(request: NextRequest, path: string[]) {
  const target = `${API_BASE_URL}/${path.join("/")}${request.nextUrl.search}`;
  const body =
    request.method === "GET" || request.method === "HEAD"
      ? undefined
      : await request.text();

  try {
    const response = await fetch(target, {
      method: request.method,
      headers: {
        "Content-Type": request.headers.get("Content-Type") ?? "application/json",
      },
      body,
      cache: "no-store",
    });
    const text = await response.text();

    return new NextResponse(text, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("Content-Type") ?? "application/json",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        detail:
          error instanceof Error
            ? error.message
            : "No se pudo conectar con la API local de ML.",
      },
      { status: 503 },
    );
  }
}

export async function GET(
  request: NextRequest,
  context: RouteContext<"/api/ml/[...path]">,
) {
  const { path } = await context.params;
  return proxy(request, path);
}

export async function POST(
  request: NextRequest,
  context: RouteContext<"/api/ml/[...path]">,
) {
  const { path } = await context.params;
  return proxy(request, path);
}
