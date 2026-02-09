# app.py
import os
import datetime as dt
from typing import Optional, Tuple, Dict, Any, List

import requests
import pandas as pd
import streamlit as st


# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


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
            "q": f"{city},KR",  # ⭐ 한국 도시 확실하게
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
                breed_part = tail.split("/")[0]  # e.g., hound-afghan
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

            # output 배열 조립 (버전 대비)
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
    habits: Dict[str, bool],
    mood: int,
    coach_style: str,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Tuple[str, str]],
    openai_api_key: str,
) -> str:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
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

    system_prompt = (
        f"{style_prompts.get(coach_style, style_prompts['따뜻한 멘토'])}\n\n"
        "출력은 반드시 한국어로 작성하라.\n"
        "아래 형식을 반드시 그대로 지켜라.\n\n"
        "형식:\n"
        "컨디션 등급: <S|A|B|C|D>\n"
        "습관 분석:\n"
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
        f"- 오늘 달성률: {rate}%\n"
        f"- 달성한 습관: {', '.join(checked) if checked else '없음'}\n"
        f"- 놓친 습관: {', '.join(missed) if missed else '없음'}\n"
        f"- 기분(1~10): {mood}\n"
        f"- 날씨: {w_txt}\n"
        f"- 오늘의 강아지 품종: {dog_breed}\n\n"
        "요구사항:\n"
        "- 컨디션 등급은 데이터에 근거해 현실적으로 부여해라.\n"
        "- 내일 미션은 실행 가능하고 구체적으로 3개.\n"
    )

    model = "gpt-5-mini"
    out = _call_openai_report(openai_api_key, model, system_prompt, user_prompt)

    if out:
        return out

    # 폴백(키 없거나 호출 실패 시)
    return (
        "컨디션 등급: C\n"
        "습관 분석:\n"
        f"- 달성률은 {rate}% 입니다. (달성: {len(checked)}개)\n"
        "- 오늘은 최소 1~2개 습관을 확실히 유지하는 전략이 좋아요.\n"
        "날씨 코멘트:\n"
        "- 날씨 정보를 가져오지 못했어요. (API Key/네트워크 확인)\n"
        "내일 미션:\n"
        "1) 물 1컵 + 5분 스트레칭\n"
        "2) 20분 집중(공부/독서)\n"
        "3) 취침 전 스크린 10분 줄이기\n"
        "오늘의 한마디:\n"
        "\"작게 해도 된다. 대신 매일 해라.\"\n"
    )


# =========================
# Session State Init
# =========================
HABITS = {
    "☀️ 기상 미션": False,
    "💧 물 마시기": False,
    "📚 공부/독서": False,
    "🏃 운동하기": False,
    "😴 수면": False,
}

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
            }
        )

    st.session_state.history = sample

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "last_weather" not in st.session_state:
    st.session_state.last_weather = None

if "last_dog" not in st.session_state:
    st.session_state.last_dog = None


# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    weather_api_key = st.text_input(
        "OpenWeatherMap API Key", type="password", value=os.getenv("OPENWEATHERMAP_API_KEY", "")
    )

    st.divider()
    debug_mode = st.checkbox("🛠 디버그 모드", value=False)
    st.caption("디버그 모드에서는 API 응답 상태를 화면에 출력합니다.")


# =========================
# Main UI
# =========================
st.title("📊 AI 습관 트래커")
st.caption("오늘 체크인 → 달성률/기분 → 날씨/강아지 → AI 코치 리포트까지!")


# =========================
# Habit Check-in UI
# =========================
st.subheader("✅ 오늘 습관 체크인")

colA, colB = st.columns([1.3, 1.0], gap="large")

with colA:
    c1, c2 = st.columns(2, gap="medium")

    habit_state = {}
    habit_keys = list(HABITS.keys())

    with c1:
        habit_state[habit_keys[0]] = st.checkbox(habit_keys[0], value=False)
        habit_state[habit_keys[1]] = st.checkbox(habit_keys[1], value=False)
        habit_state[habit_keys[2]] = st.checkbox(habit_keys[2], value=False)

    with c2:
        habit_state[habit_keys[3]] = st.checkbox(habit_keys[3], value=False)
        habit_state[habit_keys[4]] = st.checkbox(habit_keys[4], value=False)

    mood = st.slider("🙂 기분 슬라이더 (1~10)", min_value=1, max_value=10, value=7, step=1)

with colB:
    cities = [
        "Seoul",
        "Busan",
        "Incheon",
        "Daegu",
        "Daejeon",
        "Gwangju",
        "Suwon",
        "Ulsan",
        "Jeju",
        "Sejong",
    ]
    city = st.selectbox("🌍 도시 선택", options=cities, index=0)

    coach_style = st.radio(
        "🧠 코치 스타일",
        options=["스파르타 코치", "따뜻한 멘토", "게임 마스터"],
        index=1,
    )


# =========================
# Metrics + Chart
# =========================
st.subheader("📈 달성률 + 차트")

checked_cnt_now = sum(bool(v) for v in habit_state.values())
achievement_now = round((checked_cnt_now / 5) * 100)

m1, m2, m3 = st.columns(3, gap="medium")
m1.metric("달성률", f"{achievement_now}%")
m2.metric("달성 습관", f"{checked_cnt_now}/5")
m3.metric("기분", f"{mood}/10")

# 7일 데이터 만들기 (샘플 6일 + 오늘)
today_str = dt.date.today().isoformat()
hist_map = {r["date"]: r for r in st.session_state.history if "date" in r}

seven_days = []
for i in range(6, -1, -1):
    d = (dt.date.today() - dt.timedelta(days=i)).isoformat()

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
        if d == today_str:
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
# Report Generation Button
# =========================
st.subheader("📝 AI 코치 리포트")

btn = st.button("🚀 컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    # 오늘 데이터 저장
    checked_cnt = sum(bool(v) for v in habit_state.values())
    achievement = round((checked_cnt / 5) * 100)

    updated = False
    for row in st.session_state.history:
        if row.get("date") == today_str:
            row.update({"achievement": achievement, "checked": checked_cnt, "mood": mood})
            updated = True
            break

    if not updated:
        st.session_state.history.append({"date": today_str, "achievement": achievement, "checked": checked_cnt, "mood": mood})

    # API 호출
    weather = get_weather(city, weather_api_key) if weather_api_key else None
    dog = get_dog_image()

    st.session_state.last_weather = weather
    st.session_state.last_dog = dog

    # 디버그
    if debug_mode:
        st.write("🌦 Weather Raw:", weather)
        st.write("🐶 Dog Raw:", dog)

    # OpenAI 리포트 생성
    report = generate_report(
        habits=habit_state,
        mood=mood,
        coach_style=coach_style,
        weather=weather,
        dog=dog,
        openai_api_key=openai_api_key,
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
        f"- 날짜: {dt.date.today().isoformat()}\n"
        f"- 달성률: {achievement_now}% ({checked_cnt_now}/5)\n"
        f"- 기분: {mood}/10\n"
        f"- 도시: {city}\n"
        f"- 코치 스타일: {coach_style}\n\n"
        f"{report}\n"
    )

    st.markdown("#### 🔗 공유용 텍스트")
    st.code(share_text, language="text")


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

- **디버그 모드**
  - 사이드바에서 켜면 날씨/강아지 API 결과가 화면에 그대로 출력됩니다.
"""
    )
