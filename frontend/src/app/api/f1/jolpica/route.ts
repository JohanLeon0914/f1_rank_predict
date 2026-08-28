import { NextRequest, NextResponse } from "next/server";

const JOLPICA_BASE_URL = "https://api.jolpi.ca";

export async function GET(request: NextRequest) {
  const route = request.nextUrl.searchParams.get("route") ?? "/ergast/f1/current.json";

  try {
    const response = await fetch(`${JOLPICA_BASE_URL}${route}`, {
      headers: {
        "User-Agent": "F1MLPredicts/0.1.0 NextJS",
      },
      cache: "no-store",
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        detail:
          error instanceof Error
            ? error.message
            : "No se pudo consultar Jolpica.",
      },
      { status: 502 },
    );
  }
}
