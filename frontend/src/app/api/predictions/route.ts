import { createClient } from "@supabase/supabase-js";
import { NextRequest, NextResponse } from "next/server";
import type { SavedPrediction } from "@/lib/types";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey =
  process.env.SUPABASE_SECRET_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

function getSupabase() {
  if (!supabaseUrl || !supabaseKey) return null;
  return createClient(supabaseUrl, supabaseKey, {
    auth: { persistSession: false },
  });
}

type SavedPredictionPayload = SavedPrediction & {
  guest_id?: string | null;
};

export async function GET() {
  const supabase = getSupabase();
  if (!supabase) return new NextResponse(null, { status: 204 });

  const { data, error } = await supabase
    .from("predictions")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(100);

  if (error) {
    return NextResponse.json({ detail: error.message }, { status: 500 });
  }

  const predictions: SavedPrediction[] = (data ?? []).map((row) => ({
    id: row.id,
    created_at: row.created_at,
    source: row.source,
    simulation_count: row.simulation_count,
    race: {
      raceId: row.race_id,
      circuitId: row.circuit_id,
      name: row.race_name,
      date: row.race_date,
    },
    request: row.request_payload,
    averaged_predictions: row.averaged_predictions,
    raw_predictions: row.raw_predictions,
  }));

  return NextResponse.json(predictions);
}

export async function POST(request: NextRequest) {
  const supabase = getSupabase();
  if (!supabase) return new NextResponse(null, { status: 204 });

  const prediction = (await request.json()) as SavedPredictionPayload;
  const { error } = await supabase.from("predictions").insert({
    id: prediction.id,
    guest_id: prediction.guest_id,
    source: prediction.source,
    race_id: prediction.race.raceId,
    circuit_id: prediction.race.circuitId,
    race_name: prediction.race.name,
    race_date: prediction.race.date,
    simulation_count: prediction.simulation_count,
    request_payload: prediction.request,
    averaged_predictions: prediction.averaged_predictions,
    raw_predictions: prediction.raw_predictions,
  });

  if (error) {
    return NextResponse.json({ detail: error.message }, { status: 500 });
  }

  return NextResponse.json({ mode: "supabase" });
}
