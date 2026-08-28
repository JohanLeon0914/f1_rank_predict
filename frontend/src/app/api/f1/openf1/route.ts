import { NextRequest, NextResponse } from "next/server";

const OPENF1_BASE_URL = "https://api.openf1.org/v1";

export async function GET(request: NextRequest) {
  const resource = request.nextUrl.searchParams.get("resource") ?? "sessions";
  const query = new URLSearchParams(request.nextUrl.searchParams);
  query.delete("resource");

  try {
    const response = await fetch(`${OPENF1_BASE_URL}/${resource}?${query}`, {
      cache: "no-store",
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        detail:
          error instanceof Error
            ? error.message
            : "No se pudo consultar OpenF1.",
      },
      { status: 502 },
    );
  }
}
