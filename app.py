# app.py
import os
import json
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
            "q": city,
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

        # 품종 파싱: .../breeds/{breed}/... 또는 .../breeds/{breed-sub}/...
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
    OpenAI 호출 (가능하면 /v1/responses 사용, 실패 시 /v1/chat/completions 시도)
    실패 시 None
    """
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 1) Responses API (권장)
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
            # 다양한 SDK/버전 대비: output_text 우선
            if isinstance(data, dict):
                txt = data.get("output_text")
                if txt:
                    return txt
                # output 배열에서 텍스트 조립
                out = data.get("output")
                if isinstance(out, list):
                    chunks = []
                    for item in out:
                        content = item.get("content") if isinstance(item, dict) else None
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                    chunks.append(c.get("text", ""))
                        elif isinstance(content, str):
                            chunks.append(content)
                    joined = "\n".join([c for c in chunks if c]).strip()
                    if joined:
                        return joined
    except Exception:
        pass

    # 2) Chat Completions API (폴백)
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
                    return content
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
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달하여 리포트 생성
    모델: gpt-5-mini
    """
    style_prompts = {
        "스파르타 코치": (
            "당신은 매우 엄격하고 단호한 코치다. 변명은 받아주지 않는다. "
            "짧고 명확하게, 하지만 구체적인 액션을 강하게 요구하라. "
            "불필요한 미사여구는 금지."
        ),
        "따뜻한 멘토": (
            "당신은 따뜻하고 공감적인 멘토다. 사용자의 상황을 존중하고, "
            "자기효능감을 높이도록 부드럽게 격려하라. "
            "현실적인 작은 실천을 제안하라."
        ),
        "게임 마스터": (
            "당신은 RPG 게임 마스터다. 사용자의 하루를 퀘스트와 스탯으로 해석하라. "
            "보상/패널티, 레벨업, 다음 퀘스트를 재미있게 제시하라. "
            "과장된 설정은 가능하지만, 실행 가능해야 한다."
        ),
    }

    checked = [k for k, v in habits.items() if v]
    missed = [k for k, v in habits.items() if not v]
    rate = round((len(checked) / max(len(habits), 1)) * 100)

    w_txt = "날씨 정보 없음"
    if weather:
        w_txt = (
            f"{weather.get('city')} / {weather.get('description')} / "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) / 습도 {weather.get('humidity')}%"
        )

    dog_breed = dog[1] if dog else "알 수 없음"

    system_prompt = (
        f"{style_prompts.get(coach_style, style_prompts['따뜻한 멘토'])}\n\n"
        "출력은 반드시 한국어로 작성하라.\n"
        "아래 형식을 정확히 지켜라(제목/순서 유지).\n\n"
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
        "아래 데이터를 바탕으로 'AI 습관 트래커' 컨디션 리포트를 작성해줘.\n\n"
        f"- 오늘 달성률: {rate}%\n"
        f"- 달성한 습관: {', '.join(checked) if checked else '없음'}\n"
        f"- 놓친 습관: {', '.join(missed) if missed else '없음'}\n"
        f"- 기분(1~10): {mood}\n"
        f"- 날씨: {w_txt}\n"
        f"- 오늘의 강아지 품종: {dog_breed}\n\n"
        "주의:\n"
        "- 컨디션 등급은 데이터에 근거해 현실적으로 부여해줘.\n"
        "- '내일 미션'은 실행 가능한 수준(10~30분 단위 포함)으로 3개를 제안해줘.\n"
    )

    model = "gpt-5-mini"
    out = _call_openai_report(openai_api_key, model, system_prompt, user_prompt)
    if out:
        return out.strip()

    # 폴백(키 없거나 실패 시)
    return (
        "컨디션 등급: C\n"
        "습관 분석:\n"
        "- 현재는 API 호출이 불가해 기본 리포트를 표시하고 있어요.\n"
        f"- 달성률 {rate}% / 기분 {mood}/10 을 기반으로 내일은 1~2개 습관부터 확실히 잡아봐요.\n"
        "날씨 코멘트:\n"
        "- 날씨 정보를 가져오지 못했어요. (키/네트워크 확인)\n"
        "내일 미션:\n"
        "1) 물 1컵 + 5분 스트레칭\n"
        "2) 20분 집중(공부/독서)\n"
        "3) 취침 전 스크린 10분 줄이기\n"
        "오늘의 한마디:\n"
        "\"작게 시작해도, 매일이면 충분히 강해진다.\"\n"
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
    # 데모용 6일 샘플 데이터(어제까지) + 오늘은 UI 입력으로
    today = dt.date.today()
    sample = []
    # 간단한 패턴(랜덤 없이)으로 6일치 생성
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
        checked_cnt, mood = pattern[(6 - i) % len(pattern)]
        sample.append(
            {
                "date": d.isoformat(),
                "achievement": round(checked_cnt / 5 * 100),
                "checked": checked_cnt,
                "mood": mood,
            }
        )
    st.session_state.history = sample

if "today_saved" not in st.session_state:
    st.session_state.today_saved = False

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
    st.caption("키는 브라우저에 표시되지 않도록 마스킹됩니다.")


# =========================
# Main UI
# =========================
st.title("📊 AI 습관 트래커")
st.caption("오늘의 체크인 → 달성률/기분 → 날씨/강아지 → AI 코치 리포트까지 한 번에!")


# --- Check-in UI ---
st.subheader("✅ 오늘 습관 체크인")

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    # 체크박스 5개를 2열로 배치
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

    mood = st.slider("🙂 지금 기분은 어때요?", min_value=1, max_value=10, value=7, step=1)

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
        horizontal=False,
    )

    # 오늘 기록 저장 버튼(선택)
    if st.button("💾 오늘 체크인 기록 저장", use_container_width=True):
        checked_cnt = sum(bool(v) for v in habit_state.values())
        achievement = round((checked_cnt / 5) * 100)

        today = dt.date.today().isoformat()
        # 같은 날짜가 있으면 업데이트
        updated = False
        for row in st.session_state.history:
            if row.get("date") == today:
                row.update({"achievement": achievement, "checked": checked_cnt, "mood": mood})
                updated = True
                break
        if not updated:
            st.session_state.history.append(
                {"date": today, "achievement": achievement, "checked": checked_cnt, "mood": mood}
            )

        st.session_state.today_saved = True
        st.success("오늘 기록을 저장했어요! (세션 유지 동안 보관)")


# --- Metrics + Chart ---
st.subheader("📈 달성률 & 7일 추이")

checked_cnt_now = sum(bool(v) for v in habit_state.values())
achievement_now = round((checked_cnt_now / 5) * 100)

m1, m2, m3 = st.columns(3, gap="medium")
m1.metric("달성률", f"{achievement_now}%", delta=None)
m2.metric("달성 습관", f"{checked_cnt_now}/5", delta=None)
m3.metric("기분", f"{mood}/10", delta=None)

# 7일 데이터(샘플 6일 + 오늘)
today_str = dt.date.today().isoformat()
hist_map = {r["date"]: r for r in st.session_state.history if "date" in r}

# 오늘 데이터가 history에 없으면, 차트에는 "오늘 입력값"으로 표시(임시)
seven_days = []
for i in range(6, -1, -1):
    d = (dt.date.today() - dt.timedelta(days=i)).isoformat()
    if d in hist_map:
        row = hist_map[d]
        seven_days.append({"date": d, "achievement": row.get("achievement", 0), "mood": row.get("mood", 0)})
    else:
        # 오늘만 임시 반영
        if d == today_str:
            seven_days.append({"date": d, "achievement": achievement_now, "mood": mood})
        else:
            seven_days.append({"date": d, "achievement": 0, "mood": 0})

df = pd.DataFrame(seven_days)
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%m/%d")

c_chart, c_table = st.columns([1.6, 1.0], gap="large")
with c_chart:
    st.bar_chart(df.set_index("date")[["achievement"]], height=280)
with c_table:
    st.dataframe(df, use_container_width=True, height=280)


# =========================
# Report Generation
# =========================
st.subheader("📝 AI 코치 컨디션 리포트")

btn = st.button("🚀 컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    # 기록도 함께 저장(사용자 편의: 버튼 누르면 자동 저장)
    checked_cnt = sum(bool(v) for v in habit_state.values())
    achievement = round((checked_cnt / 5) * 100)

    # history 업데이트
    updated = False
    for row in st.session_state.history:
        if row.get("date") == today_str:
            row.update({"achievement": achievement, "checked": checked_cnt, "mood": mood})
            updated = True
            break
    if not updated:
        st.session_state.history.append({"date": today_str, "achievement": achievement, "checked": checked_cnt, "mood": mood})
    st.session_state.today_saved = True

    # 외부 API 호출
    weather = get_weather(city, weather_api_key) if weather_api_key else None
    dog = get_dog_image()

    st.session_state.last_weather = weather
    st.session_state.last_dog = dog

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

    # 날씨+강아지 사진 카드 (2열) + AI 리포트
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### ☁️ 오늘의 날씨")
        with st.container(border=True):
            if w:
                st.write(f"**도시:** {w.get('city')}")
                st.write(f"**상태:** {w.get('description')}")
                st.write(f"**기온:** {w.get('temp_c')}°C  (체감 {w.get('feels_like_c')}°C)")
                st.write(f"**습도:** {w.get('humidity')}%")
            else:
                st.info("날씨 정보를 가져오지 못했어요. (OpenWeatherMap API Key/네트워크 확인)")

    with right:
        st.markdown("#### 🐶 오늘의 강아지")
        with st.container(border=True):
            if dog:
                img_url, breed = dog
                st.write(f"**품종:** {breed}")
                st.image(img_url, use_container_width=True)
            else:
                st.info("강아지 이미지를 가져오지 못했어요. (Dog CEO 네트워크 확인)")

    st.markdown("#### 🧠 AI 코치 리포트")
    st.markdown(report)

    # 공유용 텍스트 (st.code)
    checked_cnt_now = sum(bool(v) for v in habit_state.values())
    achievement_now = round((checked_cnt_now / 5) * 100)
    share_text = (
        f"📊 AI 습관 트래커 공유\n"
        f"- 날짜: {dt.date.today().isoformat()}\n"
        f"- 달성률: {achievement_now}% ({checked_cnt_now}/5)\n"
        f"- 기분: {mood}/10\n"
        f"- 도시: {city}\n"
        f"- 코치 스타일: {coach_style}\n"
        f"\n---\n{report}\n"
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
  - 리포트 생성에 사용됩니다. 사이드바에 입력하세요.
  - 키가 없거나 호출이 실패하면 앱은 **기본(폴백) 리포트**를 표시합니다.

- **OpenWeatherMap API Key**
  - 현재 날씨를 가져오는 데 사용됩니다.
  - `get_weather(city, api_key)`는 **한국어(lang=kr)**, **섭씨(units=metric)**로 요청합니다.

- **Dog CEO API**
  - 무료 공개 API로 랜덤 강아지 이미지를 가져옵니다.
  - `get_dog_image()`는 실패 시 `None`을 반환합니다.

- **네트워크/응답 지연**
  - 모든 외부 호출은 `timeout=10`을 사용합니다.
  - 가끔 API가 느리거나 실패할 수 있으니 다시 시도해보세요.

- **보안**
  - 키는 화면에 마스킹되지만, 공용 PC에서는 사용 후 브라우저/세션을 종료하는 것을 권장합니다.
"""
    )
