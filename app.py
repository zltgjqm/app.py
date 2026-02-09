# app.py
import os
import datetime as dt
from typing import Optional, Tuple, Dict, Any

import requests
import pandas as pd
import streamlit as st


# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# =========================
# Constants
# =========================
CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Suwon", "Ulsan", "Jeju", "Sejong"
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]

BASE_HABITS = [
    "☀️ 기상 미션",
    "💧 물 마시기",
    "📚 공부/독서",
    "🏃 운동하기",
    "😴 수면",
]

WEEKDAYS_KR = ["월", "화", "수", "목", "금", "토", "일"]


# =========================
# API Helpers
# =========================
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap 현재 날씨 조회 (한국어, 섭씨)
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{city},KR",
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        weather_desc = None
        if isinstance(data.get("weather"), list) and data["weather"]:
            weather_desc = data["weather"][0].get("description")

        main = data.get("main", {}) or {}

        return {
            "city": city,
            "description": weather_desc,
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
        }

    except Exception:
        return None


def get_dog_image() -> Optional[Tuple[str, str]]:
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()
        if data.get("status") != "success":
            return None

        img_url = data.get("message")
        if not img_url:
            return None

        breed = "Unknown"
        try:
            parts = img_url.split("/breeds/")
            if len(parts) > 1:
                tail = parts[1]
                breed_part = tail.split("/")[0]
                breed = breed_part.replace("-", " ").title()
        except Exception:
            pass

        return img_url, breed

    except Exception:
        return None


def _call_openai_report(api_key: str, model: str, system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    OpenAI 호출 (Responses API 우선, 실패 시 Chat Completions 폴백)
    실패 시 None
    """
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 1) Responses API
    try:
        url = "https://api.openai.com/v1/responses"
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            data = r.json()
            txt = data.get("output_text")
            if txt:
                return txt.strip()

            out = data.get("output")
            if isinstance(out, list):
                chunks = []
                for item in out:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                chunks.append(c.get("text", ""))
                joined = "\n".join([c for c in chunks if c]).strip()
                if joined:
                    return joined
    except Exception:
        pass

    # 2) Chat Completions 폴백
    try:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            data = r.json()
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {})
                content = msg.get("content")
                if content:
                    return content.strip()
    except Exception:
        pass

    return None


def generate_report(
    habits: dict,
    mood: int,
    coach_style: str,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Tuple[str, str]],
    openai_api_key: str,
    selected_date: str,
    weekday_tasks: list,
    weekday_done: list,
) -> str:
    """
    습관+기분+날씨+강아지+요일별 체크리스트까지 포함하여 OpenAI에 전달
    모델: gpt-5-mini
    """

    style_prompts = {
        "스파르타 코치": (
            "당신은 매우 엄격하고 단호한 코치다. 변명은 받아주지 않는다. "
            "짧고 명확하게 말하고, 반드시 실행 가능한 지시를 내린다."
        ),
        "따뜻한 멘토": (
            "당신은 따뜻하고 공감적인 멘토다. 사용자를 비난하지 말고 "
            "현실적인 작은 실천을 통해 자신감을 올려줘라."
        ),
        "게임 마스터": (
            "당신은 RPG 게임 마스터다. 사용자의 하루를 스탯과 퀘스트로 해석하고 "
            "재미있게 다음 미션을 제시하라."
        ),
    }

    checked = [k for k, v in habits.items() if v]
    missed = [k for k, v in habits.items() if not v]
    rate = round((len(checked) / max(len(habits), 1)) * 100)

    w_txt = "날씨 정보 없음"
    if weather:
        w_txt = (
            f"{weather.get('city')} / {weather.get('description')} / "
            f"{weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C) / "
            f"습도 {weather.get('humidity')}%"
        )

    dog_breed = dog[1] if dog else "알 수 없음"

    # 요일 체크리스트 정리
    weekday_total = len(weekday_tasks)
    weekday_done_cnt = len(weekday_done)
    weekday_rate = round((weekday_done_cnt / max(weekday_total, 1)) * 100) if weekday_total > 0 else 0

    system_prompt = (
        f"{style_prompts.get(coach_style, style_prompts['따뜻한 멘토'])}\n\n"
        "출력은 반드시 한국어로 작성하라.\n"
        "아래 형식을 반드시 그대로 지켜라.\n\n"
        "형식:\n"
        "컨디션 등급: <S|A|B|C|D>\n"
        "습관 분석:\n"
        "- ...\n"
        "요일 퀘스트 분석:\n"
        "- ...\n"
        "날씨 코멘트:\n"
        "- ...\n"
        "내일 미션:\n"
        "1) ...\n"
        "2) ...\n"
        "3) ...\n"
        "오늘의 한마디:\n"
        "\"...\"\n"
    )

    user_prompt = (
        "아래 데이터를 기반으로 AI 습관 트래커 리포트를 작성해줘.\n\n"
        f"- 기록 날짜: {selected_date}\n"
        f"- 오늘 달성률(기본 습관): {rate}%\n"
        f"- 달성한 습관: {', '.join(checked) if checked else '없음'}\n"
        f"- 놓친 습관: {', '.join(missed) if missed else '없음'}\n"
        f"- 기분(1~10): {mood}\n"
        f"- 요일별 퀘스트 총 {weekday_total}개 중 {weekday_done_cnt}개 완료 ({weekday_rate}%)\n"
        f"- 완료한 요일 퀘스트: {', '.join(weekday_done) if weekday_done else '없음'}\n"
        f"- 날씨: {w_txt}\n"
        f"- 오늘의 강아지 품종: {dog_breed}\n\n"
        "요구사항:\n"
        "- 컨디션 등급은 데이터에 근거해 현실적으로 부여해라.\n"
        "- 내일 미션은 실행 가능하고 구체적으로 3개.\n"
        "- 요일 퀘스트 달성 여부를 반드시 분석해라.\n"
    )

    model = "gpt-5-mini"
    out = _call_openai_report(openai_api_key, model, system_prompt, user_prompt)

    if out:
        return out

    return (
        "컨디션 등급: C\n"
        "습관 분석:\n"
        f"- 기본 습관 달성률은 {rate}% 입니다.\n"
        "요일 퀘스트 분석:\n"
        f"- 요일 퀘스트 달성률은 {weekday_rate}% 입니다.\n"
        "날씨 코멘트:\n"
        "- 날씨 정보를 가져오지 못했어요.\n"
        "내일 미션:\n"
        "1) 물 1컵 + 5분 스트레칭\n"
        "2) 20분 집중(공부/독서)\n"
        "3) 취침 전 스크린 10분 줄이기\n"
        "오늘의 한마디:\n"
        "\"작게 해도 된다. 대신 매일 해라.\"\n"
    )


# =========================
# Utility
# =========================
def date_to_weekday_kr(d: dt.date) -> str:
    return WEEKDAYS_KR[d.weekday()]


def get_record(history: list, date_str: str) -> Optional[dict]:
    for row in history:
        if row.get("date") == date_str:
            return row
    return None


def upsert_record(history: list, record: dict):
    target = get_record(history, record["date"])
    if target:
        target.update(record)
    else:
        history.append(record)


# =========================
# Session State Init
# =========================
if "history" not in st.session_state:
    today = dt.date.today()
    sample = []
    pattern = [
        (3, 6),
        (4, 7),
        (2, 5),
        (5, 8),
        (4, 6),
        (3, 7),
    ]
    for i in range(6, 0, -1):
        d = today - dt.timedelta(days=i)
        checked_cnt, mood_val = pattern[(6 - i) % len(pattern)]
        sample.append(
            {
                "date": d.isoformat(),
                "achievement": round(checked_cnt / 5 * 100),
                "checked": checked_cnt,
                "mood": mood_val,
                "city": "Seoul",
                "coach_style": "따뜻한 멘토",
                "habits": {},
                "weekday_done": [],
            }
        )
    st.session_state.history = sample

if "weekday_task_plan" not in st.session_state:
    # 요일별 기본 체크리스트 템플릿
    st.session_state.weekday_task_plan = {
        "월": ["🧹 방 정리 10분", "📩 이메일 정리"],
        "화": ["🧠 복습 20분", "🚶 15분 산책"],
        "수": ["📚 책 10페이지", "🧘 스트레칭 10분"],
        "목": ["💻 사이드 프로젝트 30분", "💧 물 2L 목표"],
        "금": ["📝 주간 회고", "📦 다음 주 계획"],
        "토": ["🏃 운동 강하게!", "🎮 휴식도 퀘스트"],
        "일": ["😴 수면 리셋", "🍽 건강한 식사"],
    }

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "last_weather" not in st.session_state:
    st.session_state.last_weather = None

if "last_dog" not in st.session_state:
    st.session_state.last_dog = None

if "last_selected_date" not in st.session_state:
    st.session_state.last_selected_date = dt.date.today()


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password", value=os.getenv("OPENWEATHERMAP_API_KEY", ""))

    st.divider()
    debug_mode = st.checkbox("🛠 디버그 모드", value=False)

    st.divider()
    st.subheader("🗓 요일별 체크리스트 설정")
    st.caption("요일별로 매일 해야 할 일을 저장해두고 자동 불러올 수 있어요.")

    selected_weekday_for_plan = st.selectbox("요일 선택", WEEKDAYS_KR)

    plan_text = st.text_area(
        f"{selected_weekday_for_plan}요일 할 일 목록 (한 줄에 하나)",
        value="\n".join(st.session_state.weekday_task_plan.get(selected_weekday_for_plan, [])),
        height=150,
    )

    if st.button("📌 요일 체크리스트 저장", use_container_width=True):
        lines = [x.strip() for x in plan_text.split("\n") if x.strip()]
        st.session_state.weekday_task_plan[selected_weekday_for_plan] = lines
        st.success(f"{selected_weekday_for_plan}요일 체크리스트 저장 완료!")


# =========================
# Main UI
# =========================
st.title("📊 AI 습관 트래커")
st.caption("오늘/어제/원하는 날짜까지 기록하고, 요일별 퀘스트도 관리하는 업그레이드 버전 😈")


# =========================
# Calendar / Date Selection
# =========================
st.subheader("📅 기록할 날짜 선택 (달력 기능)")
selected_date = st.date_input(
    "날짜를 선택하세요 (어제 기록도 가능)",
    value=st.session_state.last_selected_date,
    max_value=dt.date.today(),
)

st.session_state.last_selected_date = selected_date
selected_date_str = selected_date.isoformat()
selected_weekday = date_to_weekday_kr(selected_date)

existing = get_record(st.session_state.history, selected_date_str)

st.info(f"선택한 날짜: **{selected_date_str} ({selected_weekday}요일)**")

if existing:
    st.success("이 날짜의 기록이 이미 있습니다. 수정/업데이트할 수 있어요.")
else:
    st.warning("이 날짜의 기록이 없습니다. 지금 작성하면 저장됩니다.")


# =========================
# Habit Check-in UI
# =========================
st.subheader("✅ 습관 체크인")

colA, colB = st.columns([1.3, 1.0], gap="large")

# 기존 데이터 불러오기
default_city = existing.get("city", "Seoul") if existing else "Seoul"
default_style = existing.get("coach_style", "따뜻한 멘토") if existing else "따뜻한 멘토"
default_mood = existing.get("mood", 7) if existing else 7
default_habits = existing.get("habits", {}) if existing else {}

with colA:
    c1, c2 = st.columns(2, gap="medium")

    habit_state = {}
    with c1:
        habit_state[BASE_HABITS[0]] = st.checkbox(BASE_HABITS[0], value=default_habits.get(BASE_HABITS[0], False))
        habit_state[BASE_HABITS[1]] = st.checkbox(BASE_HABITS[1], value=default_habits.get(BASE_HABITS[1], False))
        habit_state[BASE_HABITS[2]] = st.checkbox(BASE_HABITS[2], value=default_habits.get(BASE_HABITS[2], False))

    with c2:
        habit_state[BASE_HABITS[3]] = st.checkbox(BASE_HABITS[3], value=default_habits.get(BASE_HABITS[3], False))
        habit_state[BASE_HABITS[4]] = st.checkbox(BASE_HABITS[4], value=default_habits.get(BASE_HABITS[4], False))

    mood = st.slider("🙂 기분 슬라이더 (1~10)", min_value=1, max_value=10, value=int(default_mood), step=1)

with colB:
    city = st.selectbox("🌍 도시 선택", options=CITIES, index=CITIES.index(default_city) if default_city in CITIES else 0)
    coach_style = st.radio("🧠 코치 스타일", options=COACH_STYLES, index=COACH_STYLES.index(default_style))


# =========================
# Weekday Checklist
# =========================
st.subheader("🗓 요일별 체크리스트 (매일 해야 하는 일)")

weekday_tasks = st.session_state.weekday_task_plan.get(selected_weekday, [])
existing_weekday_done = existing.get("weekday_done", []) if existing else []

if not weekday_tasks:
    st.info("이 요일에 설정된 체크리스트가 없어요. 사이드바에서 추가할 수 있습니다.")

weekday_done = []
for task in weekday_tasks:
    done = st.checkbox(f"{task}", value=(task in existing_weekday_done))
    if done:
        weekday_done.append(task)

weekday_done_cnt = len(weekday_done)
weekday_total = len(weekday_tasks)
weekday_rate = round((weekday_done_cnt / max(weekday_total, 1)) * 100) if weekday_total > 0 else 0


# =========================
# Metrics + Chart
# =========================
st.subheader("📈 달성률 + 차트")

checked_cnt_now = sum(bool(v) for v in habit_state.values())
achievement_now = round((checked_cnt_now / 5) * 100)

m1, m2, m3, m4 = st.columns(4, gap="medium")
m1.metric("달성률", f"{achievement_now}%")
m2.metric("달성 습관", f"{checked_cnt_now}/5")
m3.metric("기분", f"{mood}/10")
m4.metric("요일 퀘스트", f"{weekday_done_cnt}/{weekday_total}")

# 최근 7일 차트 (선택한 날짜 기준)
hist_map = {r["date"]: r for r in st.session_state.history if "date" in r}

seven_days = []
for i in range(6, -1, -1):
    d = (selected_date - dt.timedelta(days=i)).isoformat()
    if d in hist_map:
        row = hist_map[d]
        seven_days.append(
            {
                "date": d,
                "achievement": row.get("achievement", 0),
                "mood": row.get("mood", 0),
            }
        )
    else:
        # 선택한 날짜면 입력값으로 임시 반영
        if d == selected_date_str:
            seven_days.append({"date": d, "achievement": achievement_now, "mood": mood})
        else:
            seven_days.append({"date": d, "achievement": 0, "mood": 0})

df = pd.DataFrame(seven_days)
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%m/%d")

chart_col, table_col = st.columns([1.6, 1.0], gap="large")
with chart_col:
    st.bar_chart(df.set_index("date")[["achievement"]], height=280)

with table_col:
    st.dataframe(df, use_container_width=True, height=280)


# =========================
# Extra Feature 1: Streak
# =========================
st.subheader("🔥 추가 기능: 연속 기록 스트릭 (Streak)")

sorted_hist = sorted(st.session_state.history, key=lambda x: x["date"])
date_set = set([x["date"] for x in sorted_hist])

# streak 계산 (오늘부터 역순으로)
streak = 0
cursor = dt.date.today()
while cursor.isoformat() in date_set:
    streak += 1
    cursor = cursor - dt.timedelta(days=1)

st.metric("연속 기록 스트릭", f"{streak}일", help="오늘 포함, 연속으로 기록이 존재하는 날짜 수")


# =========================
# Extra Feature 2: Weekly Summary
# =========================
st.subheader("📌 추가 기능: 주간 요약 (선택 날짜 기준)")

week_start = selected_date - dt.timedelta(days=selected_date.weekday())  # 월요일 시작
week_dates = [(week_start + dt.timedelta(days=i)).isoformat() for i in range(7)]

week_rows = []
for d in week_dates:
    row = hist_map.get(d)
    if row:
        week_rows.append(row)

if week_rows:
    avg_ach = round(sum(r.get("achievement", 0) for r in week_rows) / len(week_rows))
    avg_mood = round(sum(r.get("mood", 0) for r in week_rows) / len(week_rows), 1)
    st.write(f"📅 주간 평균 달성률: **{avg_ach}%**")
    st.write(f"🙂 주간 평균 기분: **{avg_mood}/10**")
else:
    st.info("이번 주 기록이 아직 없어요.")


# =========================
# Save Button
# =========================
st.subheader("💾 기록 저장")

if st.button("💾 선택한 날짜 기록 저장", use_container_width=True):
    record = {
        "date": selected_date_str,
        "achievement": achievement_now,
        "checked": checked_cnt_now,
        "mood": mood,
        "city": city,
        "coach_style": coach_style,
        "habits": habit_state,
        "weekday_done": weekday_done,
    }
    upsert_record(st.session_state.history, record)
    st.success(f"{selected_date_str} 기록 저장 완료!")


# =========================
# Report Generation Button
# =========================
st.subheader("📝 AI 코치 리포트")

btn = st.button("🚀 컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    # 저장도 자동으로 수행
    record = {
        "date": selected_date_str,
        "achievement": achievement_now,
        "checked": checked_cnt_now,
        "mood": mood,
        "city": city,
        "coach_style": coach_style,
        "habits": habit_state,
        "weekday_done": weekday_done,
    }
    upsert_record(st.session_state.history, record)

    # API 호출
    weather = get_weather(city, weather_api_key) if weather_api_key else None
    dog = get_dog_image()

    st.session_state.last_weather = weather
    st.session_state.last_dog = dog

    if debug_mode:
        st.write("🌦 Weather Raw:", weather)
        st.write("🐶 Dog Raw:", dog)

    report = generate_report(
        habits=habit_state,
        mood=mood,
        coach_style=coach_style,
        weather=weather,
        dog=dog,
        openai_api_key=openai_api_key,
        selected_date=selected_date_str,
        weekday_tasks=weekday_tasks,
        weekday_done=weekday_done,
    )
    st.session_state.last_report = report


# =========================
# Results Display
# =========================
if st.session_state.last_report:
    w = st.session_state.last_weather
    dog = st.session_state.last_dog
    report = st.session_state.last_report

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### ☁️ 오늘의 날씨")
        with st.container(border=True):
            if w:
                st.write(f"**도시:** {w.get('city')}")
                st.write(f"**상태:** {w.get('description')}")
                st.write(f"**기온:** {w.get('temp_c')}°C (체감 {w.get('feels_like_c')}°C)")
                st.write(f"**습도:** {w.get('humidity')}%")
            else:
                st.warning("날씨 정보를 가져오지 못했습니다. (OpenWeatherMap API Key 확인)")

    with right:
        st.markdown("#### 🐶 오늘의 강아지")
        with st.container(border=True):
            if dog:
                img_url, breed = dog
                st.write(f"**품종:** {breed}")
                st.image(img_url, use_container_width=True)
            else:
                st.warning("강아지 이미지를 가져오지 못했습니다. (네트워크/차단 가능)")

    st.markdown("#### 🧠 AI 코치 리포트")
    st.markdown(report)

    # 공유용 텍스트
    share_text = (
        f"📊 AI 습관 트래커 공유\n"
        f"- 날짜: {selected_date_str}\n"
        f"- 달성률: {achievement_now}% ({checked_cnt_now}/5)\n"
        f"- 기분: {mood}/10\n"
        f"- 도시: {city}\n"
        f"- 코치 스타일: {coach_style}\n"
        f"- 요일 퀘스트: {weekday_done_cnt}/{weekday_total}\n\n"
        f"{report}\n"
    )

    st.markdown("#### 🔗 공유용 텍스트")
    st.code(share_text, language="text")


# =========================
# Extra Feature 3: Export / Import
# =========================
st.subheader("📦 추가 기능: 기록 내보내기 / 불러오기")

export_col, import_col = st.columns(2, gap="large")

with export_col:
    if st.button("⬇️ JSON 내보내기", use_container_width=True):
        st.download_button(
            label="📥 다운로드",
            data=pd.DataFrame(st.session_state.history).to_json(orient="records", force_ascii=False, indent=2),
            file_name="habit_history.json",
            mime="application/json",
            use_container_width=True,
        )

with import_col:
    uploaded = st.file_uploader("⬆️ JSON 업로드(복원)", type=["json"])
    if uploaded is not None:
        try:
            imported = uploaded.read().decode("utf-8")
            parsed = pd.read_json(imported)
            if "date" in parsed.columns:
                st.session_state.history = parsed.to_dict(orient="records")
                st.success("업로드 완료! 기록이 복원되었습니다.")
            else:
                st.error("올바른 JSON 형식이 아닙니다. (date 컬럼 필요)")
        except Exception as e:
            st.error(f"업로드 실패: {e}")


# =========================
# API 안내 (expander)
# =========================
with st.expander("ℹ️ API 안내 / 문제 해결"):
    st.markdown(
        """
- **OpenAI API Key**
  - AI 코치 리포트 생성에 사용됩니다.
  - 키가 없거나 호출 실패 시 기본 리포트가 출력됩니다.

- **OpenWeatherMap API Key**
  - 현재 날씨 정보를 가져옵니다.
  - 도시를 `Seoul,KR` 형태로 요청하여 한국 도시로 확실히 지정했습니다.

- **Dog CEO API**
  - 무료 공개 API로 랜덤 강아지 이미지를 가져옵니다.
  - 네트워크 환경(학교/회사)에서 외부 API가 막혀있으면 실패할 수 있습니다.

- **달력 기능**
  - 상단 날짜 선택에서 과거 날짜를 선택하면 그 날짜 기록을 새로 작성/수정할 수 있습니다.

- **요일별 체크리스트**
  - 사이드바에서 요일별 해야 할 일을 저장하면,
    날짜를 선택할 때 해당 요일 체크리스트가 자동으로 로드됩니다.

- **추가 기능**
  - 🔥 연속 기록 스트릭
  - 📌 주간 평균 요약
  - 📦 JSON 내보내기/불러오기

- **디버그 모드**
  - 사이드바에서 켜면 날씨/강아지 API 결과가 화면에 그대로 출력됩니다.
"""
    )
