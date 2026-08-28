create extension if not exists pgcrypto;

create table if not exists public.predictions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  guest_id text,
  user_id uuid references auth.users(id) on delete set null,
  source text not null check (source in ('predicts', 'races')),
  race_id integer not null,
  circuit_id integer not null,
  race_name text not null,
  race_date date,
  simulation_count integer not null check (
    simulation_count >= 1 and simulation_count <= 100
  ),
  request_payload jsonb not null,
  averaged_predictions jsonb not null,
  raw_predictions jsonb,
  model_version text,
  api_base_url text,
  client_metadata jsonb not null default '{}'::jsonb
);

create index if not exists predictions_created_at_idx
  on public.predictions (created_at desc);

create index if not exists predictions_guest_id_created_at_idx
  on public.predictions (guest_id, created_at desc);

create index if not exists predictions_user_id_created_at_idx
  on public.predictions (user_id, created_at desc);

create index if not exists predictions_race_id_created_at_idx
  on public.predictions (race_id, created_at desc);

alter table public.predictions enable row level security;

grant select, insert on public.predictions to anon, authenticated;

drop policy if exists "guest can insert predictions" on public.predictions;
create policy "guest can insert predictions"
  on public.predictions
  for insert
  to anon, authenticated
  with check (
    simulation_count between 1 and 100
    and source in ('predicts', 'races')
  );

drop policy if exists "guest can read predictions" on public.predictions;
create policy "guest can read predictions"
  on public.predictions
  for select
  to anon, authenticated
  using (true);

drop policy if exists "users can read their own future predictions" on public.predictions;
create policy "users can read their own future predictions"
  on public.predictions
  for select
  to authenticated
  using (user_id = auth.uid());

comment on table public.predictions is
  'Stores guest and future authenticated user F1 ML race prediction simulations.';

comment on column public.predictions.guest_id is
  'Client-generated guest identifier stored in localStorage until auth is added.';

comment on column public.predictions.user_id is
  'Reserved for future required login and paid simulations.';
