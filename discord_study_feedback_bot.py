import discord, pickle, joblib, os, datetime, shap
from discord.ext import commands
from discord import app_commands
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import asyncio
from discord.ui import View, Select, Modal, TextInput, Button
from dotenv import load_dotenv
import os
import re
import json

load_dotenv()  # .env 파일 불러오기
token = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="*", intents=intents)
    
# 전역 변수
user_profiles_file = "user_profiles.json"
user_dataset_file = "user_dataset.csv"
model_file = "rf_model.pkl"
reminder_sent_file = "reminder_sent.json"

user_profiles = {}
user_dataset = pd.DataFrame()  # 잘못된 초기화 수정
reminder_jobs = {}
reminder_sent = {}  # uid: (last_sent_date, last_sent_time)
goal_expired_sent = set()  # uid 집합, 오늘 이미 goal expired DM을 보냈는지

@bot.tree.command(name="도움말", description="명령어 설명 보기")
async def 도움말(interaction: discord.Interaction):
    embed = discord.Embed(
        title="AI 학습 루틴 피드백 봇 명령어 안내",
        description=(
            "• `/생성` : 사용자 정보 등록 (닉네임, 학점 등)\n"
            "• `/입력` : 오늘의 루틴 입력 시작\n"
            "• `/프로필` : 내 프로필 정보 확인\n"
            "• `/상태` : 최근 3일 루틴 요약 및 예측\n"
            "• `/전체상태` : 전체 입력 기간 루틴 및 예측 분석\n"
            "• `/설정` : 피드백 말투, 만점 기준 등 설정 변경\n"
            "• `/리마인더` : 월~일 21:00 리마인더 기본 설정\n"
            "• `/리마인더추가` : 원하는 요일/시간 리마인더 추가\n"
            "• `/리마인더목록` : 내 리마인더 전체 목록 보기\n"
            "• `/리마인더해제 번호` : 리마인더 삭제 (번호는 /리마인더목록 참고)\n"
            "• `/초기화` : 내 정보와 입력 데이터 초기화\n"
            "\n"
            "처음 시작하신다면 `/생성` 명령어로 등록을 진행해 주세요!"
        ),
        color=discord.Color.blue()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)

# === 파일 로드 및 저장 ===
def load_user_profiles():
    global user_profiles
    if os.path.exists(user_profiles_file):
        with open(user_profiles_file, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)
    else:
        user_profiles = {}

def save_user_profiles():
    with open(user_profiles_file, "w", encoding="utf-8") as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=2)

def load_user_dataset():
    global user_dataset
    if os.path.exists(user_dataset_file):
        user_dataset = pd.read_csv(user_dataset_file)

def save_input_row(uid, row_dict):
    global user_dataset
    row_dict = row_dict.copy()
    row_dict["uid"] = uid
    row_dict["date"] = str(datetime.date.today())

    user_dataset = pd.concat([user_dataset, pd.DataFrame([row_dict])], ignore_index=True)
    user_dataset.to_csv(user_dataset_file, index=False)

    # 추가: 입력 데이터를 별도의 파일에도 저장 (append 모드, csv)
    with open("user_input_log.csv", "a", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row_dict)

def load_reminder_sent():
    global reminder_sent
    if os.path.exists(reminder_sent_file):
        with open(reminder_sent_file, "r", encoding="utf-8") as f:
            reminder_sent = json.load(f)
    else:
        reminder_sent = {}

def save_reminder_sent():
    with open(reminder_sent_file, "w", encoding="utf-8") as f:
        json.dump(reminder_sent, f, ensure_ascii=False, indent=2)

# === 모델 로드 ===
with open(model_file, "rb") as f:
    model = pickle.load(f)

# === 슬래시 명령어 등록 ===
@bot.event
async def on_ready():
    load_user_profiles()
    load_user_dataset()
    load_reminder_sent()  # 리마인더 전송 기록 불러오기
    try:
        guild = discord.Object(id=866849631878905886)
        synced = await bot.tree.sync()
        print(f"Slash commands synced: {len(synced)}개")
        for cmd in synced:
            print(f"  • /{cmd.name} - {cmd.description}")
    except Exception as e:
        print(f"Slash command sync error: {e}")
        pass
    print(f"{bot.user.name} 작동 시작!")
    # 활동 상태 설정
    activity = discord.Activity(type=discord.ActivityType.competing, name="/생성을 사용하여 시작해보세요!") # playing(하는 중), streaming(스트리밍 중), listening(듣는 중), watching(보고 있는 중), competing(참가 중), custom(사용자 정의)
    await bot.change_presence(activity=activity)
    # 리마인더 스케줄러 시작
    bot.loop.create_task(reminder_scheduler())

# === /생성 명령어: 사용자 정보 등록 ===
class RetryRegisterView(View):
    def __init__(self, uid):
        super().__init__(timeout=None)
        self.uid = uid
        # add_item 제거, 버튼은 데코레이터로만 정의

    @discord.ui.button(label="다시 입력", style=discord.ButtonStyle.danger, custom_id="retry_register")
    async def retry(self, interaction: discord.Interaction, button: Button):
        await interaction.response.send_modal(RegisterModal(self.uid))

class RetryNicknameButtonView(View):
    def __init__(self, uid):
        super().__init__(timeout=None)
        self.uid = uid

    @discord.ui.button(label="닉네임 다시 입력", style=discord.ButtonStyle.danger, custom_id="retry_nickname")
    async def retry_nickname(self, interaction: discord.Interaction, button: Button):
        await interaction.response.send_modal(RegisterModal(self.uid))

class RegisterModal(Modal):
    def __init__(self, uid):
        super().__init__(title="사용자 정보 등록")
        self.uid = uid
        self.nickname = TextInput(label="닉네임", placeholder="예: 홍길동")
        self.previous_gpa = TextInput(label="이전 학점", placeholder="예: 3.2")
        self.goal_gpa = TextInput(label="목표 학점", placeholder="예: 4.0")
        self.goal_date = TextInput(label="목표일 (YYYY-MM-DD)", placeholder="예: 2025-12-01")
        self.max_gpa = TextInput(label="학점 만점 기준", placeholder="예: 4.5")

        self.add_item(self.nickname)
        self.add_item(self.previous_gpa)
        self.add_item(self.goal_gpa)
        self.add_item(self.goal_date)
        self.add_item(self.max_gpa)

    async def on_submit(self, interaction):
        try:
            # 닉네임 중복 체크 (자기 자신 제외)
            for k, v in user_profiles.items():
                if v.get("nickname", "") == self.nickname.value and k != self.uid:
                    await interaction.response.send_message(
                        "이미 사용 중인 닉네임입니다. 다른 닉네임을 입력해주세요.",
                        view=RetryNicknameButtonView(self.uid),
                        ephemeral=True
                    )
                    return
            prev = float(self.previous_gpa.value)
            goal = float(self.goal_gpa.value)
            max_score = float(self.max_gpa.value)
            date = self.goal_date.value

            if goal > max_score or prev > max_score or goal < 0 or prev < 0:
                raise ValueError("학점 범위 오류")

            if datetime.date.fromisoformat(date) < datetime.date.today():
                raise ValueError("날짜 오류")

            user_profiles[self.uid] = {
                'nickname': self.nickname.value,
                'previous_gpa': prev,
                'goal_gpa': goal,
                'goal_date': date,
                'max_gpa': max_score,
                'tone': '친근함',
                'reminder': [{"days": ["월", "화", "수", "목", "금", "토", "일"], "time": "21:00"}]
            }
            save_user_profiles()
            await interaction.response.send_message(f"✅ 등록 완료! 환영해요 {self.nickname.value}님!", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(
                f"❌ 입력 오류가 발생했어요. 다시 입력을 원하면 아래 버튼을 눌러주세요.\n({e})",
                view=RetryRegisterView(self.uid),
                ephemeral=True
            )

@bot.tree.command(name="생성", description="사용자 정보 등록")
async def 생성(interaction: discord.Interaction):
    global user_profiles
    uid = str(interaction.user.id)
    if uid in user_profiles:
        await interaction.response.send_message("이미 등록된 사용자입니다. 수정은 `/설정` 명령어를 이용해주세요.", ephemeral=True)
    else:
        await interaction.response.send_modal(RegisterModal(uid))

# === Step1 Modal ===
class Step1Modal(Modal):
    def __init__(self, uid, dday_msg=""):
        title = "Step 1 - 스크린"
        if dday_msg:
            title += f" ({dday_msg})"
        super().__init__(title=title)
        self.uid = uid
        self.screen_study = TextInput(label="공부 화면 시간(시간)", placeholder="예: 2.5")
        self.netflix_hours = TextInput(label="OTT 시청 시간(시간)", placeholder="예: 1.0")
        self.social_media_hours = TextInput(label="SNS 사용 시간(시간)", placeholder="예: 1.5")
        self.add_item(self.screen_study)
        self.add_item(self.netflix_hours)
        self.add_item(self.social_media_hours)

    async def on_submit(self, interaction):
        def clean_float(val):
            return float(str(val).strip().rstrip('.').replace(',', ''))
        interaction.client.step_data[self.uid] = {
            'screen_study': clean_float(self.screen_study.value),
            'netflix_hours': clean_float(self.netflix_hours.value),
            'social_media_hours': clean_float(self.social_media_hours.value)
        }
        await interaction.response.send_message("Step 1 저장 완료! `다음 입력!` 버튼을 눌러주세요.", ephemeral=True, view=InputView(self.uid, step=1))

# === Step2 Modal ===
class Step2Modal(Modal):
    def __init__(self, uid):
        super().__init__(title="Step 2 - 멘탈 & 수면")
        self.uid = uid
        self.mental_health_rating = TextInput(label="정신 건강 점수 (1~10)", placeholder="예: 정신 건강이 좋을수록 높음 8")
        self.stress_level = TextInput(label="스트레스 점수 (1~10)", placeholder="예: 스트레스 낮을수록 낮음 3")
        self.sleep_hours = TextInput(label="수면 시간(시간)", placeholder="예: 7")
        self.add_item(self.mental_health_rating)
        self.add_item(self.stress_level)
        self.add_item(self.sleep_hours)

    async def on_submit(self, interaction):
        interaction.client.step_data[self.uid].update({
            'mental_health_rating': int(self.mental_health_rating.value),
            'stress_level': int(self.stress_level.value),
            'sleep_hours': float(self.sleep_hours.value)
        })
        await interaction.response.send_message("Step 2 저장 완료! `다음 입력!` 버튼을 눌러주세요.", ephemeral=True, view=InputView(self.uid, step=2))

# === Step3 Modal ===
class Step3Modal(Modal):
    def __init__(self, uid):
        super().__init__(title="Step 3 - 공부 & 시간")
        self.uid = uid
        self.time_management_score = TextInput(label="시간관리 점수 (1~10)", placeholder="예: 잘관리할 수록 높음 10")
        self.study_hours_per_day = TextInput(label="공부 시간(시간)", placeholder="예: 3")
        self.attendance_percentage = TextInput(label="출석률 (%)", placeholder="예: 100")
        self.add_item(self.time_management_score)
        self.add_item(self.study_hours_per_day)
        self.add_item(self.attendance_percentage)

    async def on_submit(self, interaction):
        interaction.client.step_data[self.uid].update({
            'time_management_score': int(self.time_management_score.value),
            'study_hours_per_day': float(self.study_hours_per_day.value),
            'attendance_percentage': float(self.attendance_percentage.value)
        })
        await interaction.response.send_message("모든 단계 완료! `오늘 결과 보기!` 버튼을 눌러 결과를 확인하세요.", ephemeral=True, view=InputView(self.uid, step=3))
        
# === 피드백 및 SHAP 시각화 ===
def generate_feedback(data, profile, prediction, feedback_top_n=None):
    model_features = [
        "screen_study",
        "study_hours_per_day",
        "netflix_hours",
        "social_media_hours",
        "screen_time",
        "mental_health_rating",
        "sleep_hours",
        "stress_level",
        "time_management_score",
        "previous_gpa",
        "attendance_percentage"
    ]
    df = pd.DataFrame([[data[feat] for feat in model_features]], columns=model_features)

    # SHAP 계산
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(df)[prediction]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_values = shap_values[0]
    # 길이 맞추기
    if len(shap_values) > len(model_features):
        shap_values = shap_values[:len(model_features)]
    elif len(shap_values) < len(model_features):
        shap_values = np.pad(shap_values, (0, len(model_features) - len(shap_values)), 'constant')
    shap_df = pd.DataFrame({
        "feature": model_features,
        "value": [data[feat] for feat in model_features],
        "shap": shap_values
    }).sort_values(by="shap", key=abs, ascending=False)

    tone = profile.get("tone", "친근함")
    nickname = profile.get("nickname", "사용자")

    # === 전체 라벨 해석 ===
    label_msg = {
        -1: {
            "친근함": "📉 성적이 떨어질 가능성이 있어요. 우리 같이 원인을 찾아보아요!",
            "분석적": "📉 예측 결과: 성적 하락 가능성",
            "동기부여": "📉 지금은 잠깐 어려울 수 있지만, 반드시 개선할 수 있어요!"
        },
         0: {
            "친근함": "➖ 현재 루틴은 성적이 유지되는 경향이에요. 조금만 다듬으면 더 좋아질 수 있어요!",
            "분석적": "➖ 예측 결과: 성적 유지 가능성",
            "동기부여": "➖ 지금도 나쁘지 않아요. 조금만 밀고 나가면 기회가 보일 거예요!"
        },
         1: {
            "친근함": "📈 성적이 오를 가능성이 높아요! 너무 잘하고 있어요 😊",
            "분석적": "📈 예측 결과: 성적 향상 가능성",
            "동기부여": "📈 훌륭합니다! 당신의 노력이 성과로 이어지고 있어요 🔥"
        }
    }

    # === 주요 변수 피드백 생성 ===
    # feature명 한글 매핑
    feature_kor_map = {
        "screen_study": "공부 화면 시간",
        "study_hours_per_day": "공부 시간",
        "netflix_hours": "OTT 시청 시간",
        "social_media_hours": "SNS 사용 시간",
        "screen_time": "총 스크린 시간",
        "mental_health_rating": "정신 건강 점수",
        "sleep_hours": "수면 시간",
        "stress_level": "스트레스 점수",
        "time_management_score": "시간관리 점수",
        "attendance_percentage": "출석률"
    }
    def explain_feature(feat, val, shap_val):
        direction = "높음" if val > 5 else "낮음"
        influence = "긍정적" if shap_val > 0 else "부정적"
        feat_kor = feature_kor_map.get(feat, feat)
        val_fmt = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        shap_fmt = f"{shap_val:.2f}" if isinstance(shap_val, (int, float)) else str(shap_val)
        # 주요 변수 템플릿 정의 (하드코딩 대신 동적으로 생성)
        base_templates = {
            "screen_time": {
            "친근함": "`{feat_kor}`이 {val_fmt}시간으로 {influence} 영향을 줬어요. 특히 SNS나 영상 시청 시간을 줄이면 더 좋을 것 같아요!",
            "분석적": "`{feat_kor}`이 {val_fmt}로 모델에 {influence} 영향을 미쳤습니다.",
            "동기부여": "`{feat_kor}`이 {val_fmt}시간으로 영향력이 있었어요. 더 집중할 수 있도록 환경을 정비해볼까요?"
            },
            "study_hours_per_day": {
            "친근함": "`{feat_kor}`이 {val_fmt}시간이에요. 꽤 괜찮지만, 조금만 더 투자해도 좋을 것 같아요!",
            "분석적": "`{feat_kor}` = {val_fmt}, 성적에 직접적인 관련이 있습니다.",
            "동기부여": "`{feat_kor}` {val_fmt}시간! 잘하고 있어요. 조금씩 늘려보면 더 좋아질 거예요!"
            },
            "sleep_hours": {
            "친근함": "`{feat_kor}`이 {val_fmt}시간인데요, 충분한 휴식이 학습 효율에 중요하다는 점, 잊지 마세요!",
            "분석적": "`{feat_kor}`는 {val_fmt}이며 {influence} 방향으로 작용 중입니다.",
            "동기부여": "`{val_fmt}`시간의 `{feat_kor}`! 에너지를 더 채우면 내일 더 멋진 성과가 기다릴 거예요!"
            },
            "mental_health_rating": {
            "친근함": "`{feat_kor}`이 {val_fmt}/10이에요. 정신적으로 안정되면 학습에도 큰 도움이 돼요!",
            "분석적": "`{feat_kor}` = {val_fmt} → 모델에서 중요한 정서적 요소로 작용.",
            "동기부여": "멘탈 점수 {val_fmt}! 당신의 마음 상태도 소중해요. 건강한 마음으로 앞으로 나아가요!"
            },
            "time_management_score": {
            "친근함": "`{feat_kor}`가 {val_fmt}/10이에요. 계획적인 하루는 더 나은 결과를 만들어요!",
            "분석적": "`{feat_kor}` = {val_fmt} → 루틴 구조와 관련한 변수.",
            "동기부여": "{val_fmt}/10의 `{feat_kor}`! 목표가 분명해 보입니다. 멋져요!"
            }
        }
        # 나머지 변수는 공통 템플릿 사용
        templates = {}
        for feat in model_features:
            if feat in base_templates:
                templates[feat] = base_templates[feat]
            else:
                templates[feat] = {
                    "친근함": f"`{{feat_kor}}` 값이 {{val_fmt}}이라서 {{influence}} 영향이 있었어요!",
                    "분석적": f"`{{feat_kor}}` = {{val_fmt}} → {{influence}} 영향.",
                    "동기부여": f"`{{feat_kor}}`(현재 {{val_fmt}})이 결과에 영향을 주었어요. 좋은 방향으로 바꿔볼 수 있어요!"
                }
        # 템플릿 포맷팅
        for feat in templates:
            for tone_key in templates[feat]:
                templates[feat][tone_key] = templates[feat][tone_key].format(
                feat_kor=feat_kor, val_fmt=val_fmt, influence=influence
            )
        if feat in templates:
            return templates[feat][tone]
        else:
            # fallback
            return {
                "친근함": f"`{feat_kor}` 값이 {val_fmt}이라서 {influence} 영향이 있었어요!",
                "분석적": f"`{feat_kor}` = {val_fmt} → {influence} 영향.",
                "동기부여": f"`{feat_kor}`(현재 {val_fmt})이 결과에 영향을 주었어요. 좋은 방향으로 바꿔볼 수 있어요!"
            }[tone]

    # 피드백에서 이전학점, 출석률 제외 (상위 feature 선정 시)
    feedback_features_df = shap_df[~shap_df["feature"].isin(["previous_gpa", "attendance_percentage"])]
    if tone == "친근함":
        feedback_rows = feedback_features_df.head(3).iterrows()
    elif tone == "동기부여":
        feedback_rows = feedback_features_df.head(5).iterrows()
    else:  # 분석적
        feedback_rows = feedback_features_df.iterrows()
    detail_lines = [explain_feature(row.feature, row.value, row.shap) for _, row in feedback_rows]
    final_feedback = label_msg[prediction][tone] + "\n\n" + "\n".join(f"• {line}" for line in detail_lines)
    return final_feedback, shap_df

# === SHAP 시각화 함수 ===
def generate_shap_plot(shap_df):
    shap_df = shap_df.copy()
    
    # 영어 feature명으로 변환
    feature_name_map = {
        "screen_study": "Screen Study",
        "study_hours_per_day": "Study Hours",
        "netflix_hours": "OTT Hours",
        "social_media_hours": "SNS",
        "screen_time": "Total Screen",
        "mental_health_rating": "Mental Health",
        "sleep_hours": "Sleep",
        "stress_level": "Stress",
        "time_management_score": "Time Management"
    }

    # 시각화 대상만 추출 (이전 학점, 출석률 제외)
    shap_df = shap_df[~shap_df["feature"].isin(["previous_gpa", "attendance_percentage"])]
    
    # 이름 변환
    shap_df["feature"] = shap_df["feature"].map(feature_name_map)
    
    # SHAP 값 스케일링 (정규화: -1 ~ 1)
    max_val = shap_df["shap"].abs().max()
    shap_df["scaled_shap"] = shap_df["shap"] / max_val if max_val != 0 else shap_df["shap"]
    
    # 시각화 순서 역순으로 정렬
    shap_df = shap_df.sort_values("scaled_shap")
    
    # 색상 설정
    colors = ["#4caf50" if val > 0 else "#f44336" for val in shap_df["scaled_shap"]]

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(6, max(4, 0.5 * len(shap_df))))
    ax.barh(shap_df["feature"], shap_df["scaled_shap"], color=colors)

    # SHAP 값 숫자 표기
    for i, (value, name) in enumerate(zip(shap_df["shap"], shap_df["feature"])):
        ax.text(
            value / max_val + 0.02 if value > 0 else value / max_val - 0.02,
            i,
            f"{value:.3f}",
            va="center",
            ha="left" if value > 0 else "right",
            fontsize=8
        )

    # 스타일 설정
    ax.set_title("Feature Importance (SHAP)", fontsize=12)
    ax.set_xlabel("Normalized SHAP Value (Impact)", fontsize=10)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    # 결과를 이미지로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# 전역 모델 변수
rf_model = joblib.load("rf_model.pkl")  # ⬅️ model → rf_model로 명확하게 이름 변경

# === 오늘 결과 보기 View ===
class InputView(View):
    def __init__(self, uid, step):
        # timeout=None로 해도 interaction 만료는 발생할 수 있음(Discord 제한)
        # interaction이 만료되면 버튼을 눌러도 콜백이 아예 호출되지 않으므로,
        # interaction이 만료되기 전에 버튼을 눌러야만 동작함.
        # Discord의 interaction 만료(15분) 이후에는 버튼이 비활성화됨(회색).
        # 이를 완전히 해결하는 방법은 없음(Discord 정책).
        # 단, 버튼이 눌릴 때마다 view를 새로 보내서 interaction을 갱신하는 방식으로 UX를 개선할 수 있음.
        super().__init__(timeout=None)
        self.uid = uid
        self.step = step
        if step == 1 or step == 2:
            next_btn = Button(label="다음 입력!", style=discord.ButtonStyle.primary, custom_id="next_step")
            next_btn.callback = self.next_step_callback
            self.add_item(next_btn)
        elif step == 3:
            result_btn = Button(label="오늘 결과 보기!", style=discord.ButtonStyle.success, custom_id="show_result")
            result_btn.callback = self.show_result_callback
            self.add_item(result_btn)

    async def next_step_callback(self, interaction: discord.Interaction):
        steps = interaction.client.step_status.get(self.uid, 0)
        if steps == 0:
            await interaction.response.send_modal(Step2Modal(self.uid))
            interaction.client.step_status[self.uid] = 1
        elif steps == 1:
            await interaction.response.send_modal(Step3Modal(self.uid))
            interaction.client.step_status[self.uid] = 2

        # 버튼을 다시 보내 interaction을 갱신(만료 방지 UX)
        await interaction.followup.send(
            "다음 입력 단계로 이동했습니다.",
            ephemeral=True,
            view=InputView(self.uid, step=steps+1)
        )

    async def show_result_callback(self, interaction: discord.Interaction):
        data = interaction.client.step_data[self.uid]
        profile = user_profiles[self.uid]
        nickname = self.uid

        # 전처리
        data['screen_time'] = data['screen_study'] + data['netflix_hours'] + data['social_media_hours']
        data['previous_gpa'] = profile["previous_gpa"] / profile["max_gpa"] * 4.0  # <-- 예측 모델에 맞게 4.0 만점으로 스케일링

        # 예측 입력용 순서 고정
        model_features = [
            "screen_study",
            "study_hours_per_day",
            "netflix_hours",
            "social_media_hours",
            "screen_time",
            "mental_health_rating",
            "sleep_hours",
            "stress_level",
            "time_management_score",
            "previous_gpa",
            "attendance_percentage"
        ]
        df = pd.DataFrame([[data[feat] for feat in model_features]], columns=model_features)

        # 예측
        prediction = rf_model.predict(df)[0]

        # 피드백 및 시각화
        feedback, shap_df = generate_feedback(data, profile, prediction)
        plot_buf = generate_shap_plot(shap_df)

        # 저장
        save_input_row(nickname, data)

        # 응답
        file = discord.File(plot_buf, filename="shap.png")
        try:
            await interaction.response.send_message(content=feedback, file=file, ephemeral=True)
        except discord.errors.NotFound:
            await interaction.followup.send(content=feedback, file=file, ephemeral=True)
        except discord.InteractionResponded:
            await interaction.followup.send(content=feedback, file=file, ephemeral=True)

# === 목표일 성적 입력 Modal ===
class GoalGpaInputModal(Modal):
    def __init__(self, uid):
        super().__init__(title="목표일 성적 입력")
        self.uid = uid
        self.final_gpa = TextInput(label="최종 학점", placeholder="예: 3.8")
        self.add_item(self.final_gpa)

    async def on_submit(self, interaction):
        uid = self.uid
        try:
            gpa = float(self.final_gpa.value)
            user_profiles[uid]["final_gpa"] = gpa
            save_user_profiles()
            # 전체 데이터셋에서 평균값 계산 및 종합 피드백
            import csv
            msg = f"🎉 목표일에 도달했습니다!\n최종 학점: {gpa}\n"
            if not user_dataset.empty and "uid" in user_dataset.columns:
                df = user_dataset[user_dataset["uid"].astype(str) == uid].copy()
                if not df.empty:
                    mean_vals = {col: df[col].mean() for col in df.columns if col not in ["uid", "date"]}
                    mean_vals["uid"] = uid
                    mean_vals["final_gpa"] = gpa
                    mean_vals["goal_date"] = user_profiles[uid]["goal_date"]
                    msg += "\n[전체 기간 평균 루틴]\n"
                    for k, v in mean_vals.items():
                        if k not in ["uid", "goal_date", "final_gpa"]:
                            msg += f"- {k}: {v:.2f}\n"
                    # user_total_dataset.csv에 append
                    file_exists = os.path.exists("user_total_dataset.csv")
                    with open("user_total_dataset.csv", "a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(mean_vals.keys()))
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(mean_vals)
            await interaction.response.send_message(msg, ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"입력 오류: {e}", ephemeral=True)

@bot.tree.command(name="입력수정", description="오늘 입력을 삭제하고 다시 입력")
async def 입력수정(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    today = str(datetime.date.today())
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
        return
    # 오늘 입력 데이터 삭제
    global user_dataset
    if not user_dataset.empty and "uid" in user_dataset.columns and "date" in user_dataset.columns:
        mask = ~((user_dataset["uid"].astype(str) == uid) & (user_dataset["date"].astype(str) == today))
        if mask.sum() == len(user_dataset):
            await interaction.response.send_message("오늘 입력된 데이터가 없습니다.", ephemeral=True)
            return
        user_dataset = user_dataset[mask]
        user_dataset.to_csv(user_dataset_file, index=False)
    else:
        await interaction.response.send_message("오늘 입력된 데이터가 없습니다.", ephemeral=True)
        return
    await interaction.response.send_message("오늘 입력이 삭제되었습니다. 다시 입력을 시작합니다.", ephemeral=True)
    # Step1Modal은 followup으로 전송 (interaction에 두 번 응답 방지)
    await interaction.followup.send_modal(Step1Modal(uid))

@bot.tree.command(name="입력", description="오늘 루틴 입력 시작")
async def 입력(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
        return
    await start_input_flow(interaction, uid)

async def start_input_flow(interaction, uid):
    today = str(datetime.date.today())
    goal_date = user_profiles[uid].get("goal_date")
    # 오늘이 목표일이면 성적 입력 폼
    if goal_date == today:
        await interaction.response.send_modal(GoalGpaInputModal(uid))
        return
    # 오늘 이미 입력한 데이터가 있는지 uid와 date로 확인 (타입 일치 보장)
    if not user_dataset.empty and (
        (user_dataset["uid"].astype(str) == uid) & (user_dataset["date"].astype(str) == today)
    ).any():
        await interaction.response.send_message("오늘은 이미 입력하셨어요! `/상태` 명령어로 확인해보세요.", ephemeral=True)
        return
    if not hasattr(interaction.client, "step_data"):
        interaction.client.step_data = {}
        interaction.client.step_status = {}
    interaction.client.step_status[uid] = 0
    # D-day 계산
    dday = None
    if goal_date:
        try:
            dday = (datetime.date.fromisoformat(goal_date) - datetime.date.today()).days
        except:
            dday = None
    dday_msg = f"목표일까지 D-{dday}" if dday is not None and dday >= 0 else ""
    # D-day 메시지를 Step1Modal의 제목에 포함하여 한 번만 응답
    await interaction.response.send_modal(Step1Modal(uid, dday_msg))

class ReminderInputView(View):
    def __init__(self, user_id):
        super().__init__(timeout=None)
        self.user_id = user_id

    @discord.ui.button(label="입력하기!", style=discord.ButtonStyle.primary, custom_id="reminder_input")
    async def input_button(self, interaction: discord.Interaction, button: Button):
        if str(interaction.user.id) != str(self.user_id):
            await interaction.response.send_message("본인만 사용할 수 있습니다.", ephemeral=True)
            return
        await start_input_flow(interaction, str(self.user_id))
        # 버튼을 다시 보내 interaction을 갱신(만료 방지 UX)
        await interaction.followup.send(
            "입력 플로우를 시작합니다.",
            ephemeral=True,
            view=InputView(str(self.user_id), step=1)
        )
        return

async def send_reminder_dm(user_id):
    try:
        #print(f"[리마인더 DM 시도] {user_id}")
        today = str(datetime.date.today())
        if not user_dataset.empty and "uid" in user_dataset.columns and "date" in user_dataset.columns:
            if ((user_dataset["uid"].astype(str) == str(user_id)) & (user_dataset["date"].astype(str) == today)).any():
                #print(f"[리마인더 DM 스킵] {user_id}: 이미 오늘 입력함")
                return
        user = await bot.fetch_user(int(user_id))
        if user is None:
            #print(f"[리마인더 DM 실패] {user_id}: 유저를 찾을 수 없습니다.")
            return
        view = ReminderInputView(user_id)
        await user.send("⏰ 오늘의 루틴을 입력해 주세요! 아래 버튼을 눌러 입력을 시작하세요.", view=view)
        #print(f"[리마인더 DM 성공] {user_id}")
    except discord.Forbidden:
        #print(f"[리마인더 DM 실패] {user_id}: DM 권한이 없습니다. (유저가 DM 차단)")
        pass
    except Exception as e:
        #print(f"[리마인더 DM 실패] {user_id}: {e}")
        pass

async def send_goal_expired_dm(user_id):
    user = await bot.fetch_user(int(user_id))
    if user:
        try:
            await user.send("🎯 목표일이 지났습니다! 새로운 목표일을 `/설정` 명령어로 입력해 주세요.")
        except Exception as e:
            print(f"[목표일 만료 DM 실패] {user_id}: {e}")

async def reminder_scheduler():
    #print("[리마인더 스케줄러 시작]")
    await bot.wait_until_ready()
    while not bot.is_closed():
        load_user_profiles()
        load_user_dataset()
        # load_reminder_sent()  # 불필요: on_ready에서 1회만 호출
        now = datetime.datetime.now()
        weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        today = weekday_map[now.weekday()]
        now_time = now.strftime("%H:%M")
        now_date = now.date()
        # ...existing code...
        for uid, profile in user_profiles.items():
            reminders = profile.get("reminder", [])
            if not isinstance(reminders, list):
                continue
            for r in reminders:
                days = r.get("days", [])
                time = r.get("time")
                if today in days and now_time >= time:
                    # reminder_sent의 값이 (str(now_date), time)과 같으면 이미 전송한 것
                    if reminder_sent.get(uid) != [str(now_date), time]:
                        await send_reminder_dm(uid)
                        reminder_sent[uid] = [str(now_date), time]
                        save_reminder_sent()  # 전송 기록 저장
        # 목표일 알림 (매일 09:00에만 체크)
        for uid, profile in user_profiles.items():
            goal_date = profile.get("goal_date")
            if not goal_date:
                continue
            try:
                goal_dt = datetime.date.fromisoformat(goal_date)
            except:
                continue
            if now.hour == 9 and now.minute == 0:
                if goal_dt < now_date and (uid not in goal_expired_sent):
                    if (now_date - goal_dt).days == 1:
                        await send_goal_expired_dm(uid)
                        goal_expired_sent.add(uid)
        await asyncio.sleep(1)


@bot.tree.command(name="프로필", description="내 프로필 정보 보기")
async def 프로필(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
        return
    profile = user_profiles[uid]
    # 리마인더를 사람이 읽기 좋은 문자열로 변환
    reminder = profile.get('reminder', '설정 안됨')
    if isinstance(reminder, list):
        if reminder:
            reminder = ", ".join(f"요일: {', '.join(r['days'])}, 시간: {r['time']}" for r in reminder)
        else:
            reminder = '설정 안됨'
    elif isinstance(reminder, str):
        reminder = reminder.replace('\n', '').replace('\\n', '')
    msg = (
        f"**[내 프로필 정보]**\n"
        f"- 닉네임: {profile.get('nickname', '')}\n"
        f"- 이전 학점: {profile.get('previous_gpa', '')}\n"
        f"- 목표 학점: {profile.get('goal_gpa', '')}\n"
        f"- 목표일: {profile.get('goal_date', '')}\n"
        f"- 학점 만점 기준: {profile.get('max_gpa', '')}\n"
        f"- 피드백 말투: {profile.get('tone', '')}\n"
        f"- 리마인더: {reminder if reminder else '설정 안됨'}\n"
    )
    if "final_gpa" in profile:
        msg += f"- 최종 학점: {profile['final_gpa']}\n"
    await interaction.response.send_message(msg, ephemeral=True)

# === /상태 명령어 ===
@bot.tree.command(name="상태", description="최근 루틴 상태 요약")
async def 상태(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
        return
    profile = user_profiles[uid]
    if user_dataset.empty or uid not in user_dataset["uid"].astype(str).values:
        await interaction.response.send_message("아직 입력된 데이터가 없습니다. `/입력` 명령어로 입력해주세요.", ephemeral=True)
        return
    df = user_dataset[user_dataset["uid"].astype(str) == uid].copy()
    if len(df) < 3:
        await interaction.response.send_message("최근 입력이 3일 이상 필요합니다.", ephemeral=True)
        return
    df_recent = df.tail(3)
    # 최근 평균값 계산
    mean_vals = {
        "screen_study": df_recent["screen_study"].mean(),
        "study_hours_per_day": df_recent["study_hours_per_day"].mean(),
        "netflix_hours": df_recent["netflix_hours"].mean(),
        "social_media_hours": df_recent["social_media_hours"].mean(),
        "screen_time": df_recent["screen_time"].mean(),
        "mental_health_rating": df_recent["mental_health_rating"].mean(),
        "sleep_hours": df_recent["sleep_hours"].mean(),
        "stress_level": df_recent["stress_level"].mean(),
        "time_management_score": df_recent["time_management_score"].mean(),
        "previous_gpa": profile["previous_gpa"] / profile["max_gpa"] * 4.0,
        "attendance_percentage": df_recent["attendance_percentage"].mean()
    }
    # 예측
    model_features = [
        "screen_study",
        "study_hours_per_day",
        "netflix_hours",
        "social_media_hours",
        "screen_time",
        "mental_health_rating",
        "sleep_hours",
        "stress_level",
        "time_management_score",
        "previous_gpa",
        "attendance_percentage"
    ]
    df_pred = pd.DataFrame([[mean_vals[feat] for feat in model_features]], columns=model_features)
    prediction = rf_model.predict(df_pred)[0]
    feedback, shap_df = generate_feedback(mean_vals, profile, prediction)
    plot_buf = generate_shap_plot(shap_df)
    # Graph (last 3 days trend, excluding previous GPA)
    fig, ax = plt.subplots()
    plot_vars = [
        ("screen_study", "Screen Study Time"),
        ("study_hours_per_day", "Study Hours"),
        ("netflix_hours", "OTT Hours"),
        ("social_media_hours", "SNS Hours"),
        ("screen_time", "Screen Time"),
        ("mental_health_rating", "Mental Health Score"),
        ("sleep_hours", "Sleep Hours"),
        ("stress_level", "Stress Score"),
        ("time_management_score", "Time Management Score")
    ]
    for col, label in plot_vars:
        ax.plot(df_recent[col].values, label=label)
    ax.set_title(f"{profile['nickname']}'s Last 3 Days Routine Trend")
    ax.legend(fontsize=8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # 메시지
    msg = f"최근 3일 평균 루틴 기반 예측 및 피드백입니다.\n\n{feedback}"

    file1 = discord.File(buf, filename="routine_summary.png")
    file2 = discord.File(plot_buf, filename="shap.png")
    try:
        await interaction.response.send_message(content=msg, files=[file1, file2], ephemeral=True)
    except discord.errors.NotFound:
        await interaction.followup.send(content=msg, files=[file1, file2], ephemeral=True)
    except discord.InteractionResponded:
        await interaction.followup.send(content=msg, files=[file1, file2], ephemeral=True)

# === /전체상태 명령어 ===
@bot.tree.command(name="전체상태", description="전체 입력 기간 루틴 및 예측 분석")
async def 전체상태(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
        return
    profile = user_profiles[uid]
    tone = profile.get("tone", "친근함")
    if user_dataset.empty or uid not in user_dataset["uid"].astype(str).values:
        await interaction.response.send_message("아직 입력된 데이터가 없습니다. `/입력` 명령어로 입력해주세요.", ephemeral=True)
        return
    df = user_dataset[user_dataset["uid"].astype(str) == uid].copy()
    if len(df) < 1:
        await interaction.response.send_message("입력 데이터가 1개 이상 필요합니다.", ephemeral=True)
        return
    # 전체 평균값 계산
    mean_vals = {
        "screen_study": df["screen_study"].mean(),
        "study_hours_per_day": df["study_hours_per_day"].mean(),
        "netflix_hours": df["netflix_hours"].mean(),
        "social_media_hours": df["social_media_hours"].mean(),
        "screen_time": df["screen_time"].mean(),
        "mental_health_rating": df["mental_health_rating"].mean(),
        "sleep_hours": df["sleep_hours"].mean(),
        "stress_level": df["stress_level"].mean(),
        "time_management_score": df["time_management_score"].mean(),
        "previous_gpa": profile["previous_gpa"] / profile["max_gpa"] * 4.0,
        "attendance_percentage": df["attendance_percentage"].mean()
    }
    # 예측
    model_features = [
        "screen_study",
        "study_hours_per_day",
        "netflix_hours",
        "social_media_hours",
        "screen_time",
        "mental_health_rating",
        "sleep_hours",
        "stress_level",
        "time_management_score",
        "previous_gpa",
        "attendance_percentage"
    ]
    df_pred = pd.DataFrame([[mean_vals[feat] for feat in model_features]], columns=model_features)
    prediction = rf_model.predict(df_pred)[0]
    # === 피드백 변수 개수 tone별로 전달 ===
    feedback_top_n = None
    if tone == "친근함":
        feedback_top_n = 3
    elif tone == "동기부여":
        feedback_top_n = 5
    # 분석적은 None (전체)
    feedback, shap_df = generate_feedback(mean_vals, profile, prediction, feedback_top_n=feedback_top_n)
    plot_buf = generate_shap_plot(shap_df)
    # 전체 기간 트렌드 그래프 (날짜별 주요 변수, 영어 라벨)
    fig, ax = plt.subplots()
    plot_vars = [
        ("screen_study", "Screen Study Time"),
        ("study_hours_per_day", "Study Hours"),
        ("netflix_hours", "OTT Hours"),
        ("social_media_hours", "SNS Hours"),
        ("screen_time", "Screen Time"),
        ("mental_health_rating", "Mental Health Score"),
        ("sleep_hours", "Sleep Hours"),
        ("stress_level", "Stress Score"),
        ("time_management_score", "Time Management Score")
    ]
    for col, label in plot_vars:
        ax.plot(df["date"], df[col], label=label)
    ax.set_title(f"{profile['nickname']}'s Routine Trend (All)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    # 변화 자연어 피드백 생성 (tone별)
    def trend_feedback(col, label, up_good=None, down_good=None):
        first = df[col].iloc[0]
        last = df[col].iloc[-1]
        diff = last - first
        if abs(diff) < 0.05 * (abs(first) + 1e-6):
            if tone == "동기부여":
                return f"- {label}: 꾸준함이 가장 큰 힘이에요! 계속 유지해봐요."
            elif tone == "분석적":
                return f"- {label}: 큰 변화 없이 안정적으로 유지되었습니다."
            else:
                return f"- {label}: 큰 변화 없이 비슷하게 유지되고 있어요."
        if diff > 0:
            if up_good:
                return f"- {label}: {up_good} (처음 {first:.2f} → 최근 {last:.2f})"
            if tone == "동기부여":
                return f"- {label}: 점점 좋아지고 있어요! (처음 {first:.2f} → 최근 {last:.2f})"
            elif tone == "분석적":
                return f"- {label}: 수치가 증가했습니다. (처음 {first:.2f} → 최근 {last:.2f})"
            else:
                return f"- {label}: 최근에 증가했어요! (처음 {first:.2f} → 최근 {last:.2f})"
        else:
            if down_good:
                return f"- {label}: {down_good} (처음 {first:.2f} → 최근 {last:.2f})"
            if tone == "동기부여":
                return f"- {label}: 더 나은 방향으로 변화 중이에요! (처음 {first:.2f} → 최근 {last:.2f})"
            elif tone == "분석적":
                return f"- {label}: 수치가 감소했습니다. (처음 {first:.2f} → 최근 {last:.2f})"
            else:
                return f"- {label}: 최근에 감소했어요! (처음 {first:.2f} → 최근 {last:.2f})"
    trend_msgs = [
        trend_feedback("screen_study", "Screen Study Time", up_good="공부 화면 시간이 늘어나고 있어요!", down_good="공부 화면 시간이 줄고 있어요."),
        trend_feedback("study_hours_per_day", "Study Hours", up_good="공부 시간이 늘고 있습니다!", down_good="공부 시간이 줄고 있습니다."),
        trend_feedback("netflix_hours", "OTT Hours", up_good="OTT 시청 시간이 늘고 있습니다.", down_good="OTT 시청 시간이 줄고 있습니다."),
        trend_feedback("social_media_hours", "SNS Hours", up_good="SNS 사용 시간이 늘고 있습니다.", down_good="SNS 사용 시간이 줄고 있습니다."),
        trend_feedback("screen_time", "Screen Time", up_good="전체 스크린 시간이 늘고 있습니다.", down_good="전체 스크린 시간이 줄고 있습니다."),
        trend_feedback("mental_health_rating", "Mental Health Score", up_good="정신 건강 점수가 좋아지고 있습니다!", down_good="정신 건강 점수가 낮아지고 있습니다."),
        trend_feedback("sleep_hours", "Sleep Hours", up_good="수면 시간이 늘고 있습니다.", down_good="수면 시간이 줄고 있습니다."),
        trend_feedback("stress_level", "Stress Score", up_good="스트레스가 증가하고 있습니다.", down_good="스트레스가 줄고 있습니다!"),
        trend_feedback("time_management_score", "Time Management Score", up_good="시간 관리 점수가 좋아지고 있습니다!", down_good="시간 관리 점수가 낮아지고 있습니다.")
    ]
    trend_text = "\n".join(trend_msgs)
    # 메시지
    msg = f"전체 입력 기간 평균 루틴 기반 예측 및 피드백입니다.\n\n{feedback}\n\n[전체 기간 변화 요약]\n{trend_text}"
    file1 = discord.File(buf, filename="routine_all_summary.png")
    try:
        file2 = discord.File(plot_buf, filename="shap.png")
    except Exception:
        # plot_buf이 이미 닫혔거나 사용 불가한 경우, 새로 생성
        plot_buf = generate_shap_plot(shap_df)
        file2 = discord.File(plot_buf, filename="shap.png")
    try:
        await interaction.response.send_message(content=msg, files=[file1, file2], ephemeral=True)
    except Exception:
        try:
            await interaction.followup.send(content=msg, files=[file1, file2], ephemeral=True)
        except Exception:
            # 마지막 fallback: 파일 없이 텍스트만 전송
            await interaction.followup.send(content=msg, ephemeral=True)

# === /설정 명령어 ===
from discord.ui import View, Select, Modal, TextInput, Button

class ToneSelectView(View):
    def __init__(self, uid):
        super().__init__(timeout=None)
        self.uid = uid
        self.tone_select = Select(
            placeholder="피드백 말투를 선택하세요",
            options=[
                discord.SelectOption(label="친근함", value="친근함", description="따뜻하고 부드러운 피드백"),
                discord.SelectOption(label="분석적", value="분석적", description="논리적인 해설 중심"),
                discord.SelectOption(label="동기부여", value="동기부여", description="의욕을 불러일으키는 응원형 피드백")
            ],
            min_values=1,
            max_values=1
        )
        self.tone_select.callback = self.tone_selected
        self.add_item(self.tone_select)

    async def tone_selected(self, interaction: discord.Interaction):
        tone = self.tone_select.values[0]
        await interaction.response.send_modal(MaxGpaModal(self.uid, tone))
        # 버튼 갱신(UX)
        await interaction.followup.send("학점 만점 기준을 입력하세요.", ephemeral=True, view=self)

class MaxGpaModal(Modal):
    def __init__(self, uid, tone):
        super().__init__(title="학점 만점 기준 입력")
        self.uid = uid
        self.tone = tone
        self.max_gpa = TextInput(label="학점 만점 기준 (예: 4.5)", placeholder="예: 4.5")
        self.add_item(self.max_gpa)

    async def on_submit(self, interaction):
        try:
            max_score = float(self.max_gpa.value)
            if max_score <= 0 or max_score > 5.0:
                raise ValueError("유효한 학점 범위 아님")
            user_profiles[self.uid]["max_gpa"] = max_score
            user_profiles[self.uid]["tone"] = self.tone
            save_user_profiles()
            await interaction.response.send_message(f"✅ 설정이 성공적으로 저장되었습니다! (말투: {self.tone}, 만점: {max_score})", ephemeral=True)
        except:
            await interaction.response.send_message("❌ 입력이 올바르지 않습니다. 다시 시도해주세요.", ephemeral=True)

@bot.tree.command(name="설정", description="설정 변경")
async def 설정(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
    else:
        await interaction.response.send_message("피드백 말투를 선택하세요:", view=ToneSelectView(uid), ephemeral=True)

# === /리마인더 명령어 ===
@bot.tree.command(name="리마인더", description="리마인더 설정")
async def 리마인더(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
    else:
        # 입력 폼 없이 월~일 21:00으로 바로 설정
        user_profiles[uid]["reminder"] = [{"days": ["월", "화", "수", "목", "금", "토", "일"], "time": "21:00"}]
        save_user_profiles()
        await interaction.response.send_message("✅ 리마인더가 월~일 21:00으로 기본 설정되었습니다!", ephemeral=True)

# === /리마인더 추가 명령어 ===
class ReminderAddView(View):
    def __init__(self, uid):
        super().__init__(timeout=None)
        self.uid = uid
        self.day_select = Select(
            placeholder="요일 선택 (복수 선택 가능)",
            options=[discord.SelectOption(label=d, value=d) for d in ["월", "화", "수", "목", "금", "토", "일"]],
            min_values=1,
            max_values=7
        )
        self.time_select = Select(
            placeholder="시간 선택 (24시 기준)",
            options=[discord.SelectOption(label=f"{h:02d}:00", value=f"{h:02d}:00") for h in range(6, 24)],
            min_values=1,
            max_values=1
        )
        self.day_select.callback = self.day_selected
        self.time_select.callback = self.time_selected
        self.add_item(self.day_select)
        self.add_item(self.time_select)
        self.selected_days = None
        self.selected_time = None

    async def day_selected(self, interaction: discord.Interaction):
        self.selected_days = self.day_select.values
        if self.selected_time:
            await self.save_reminder(interaction)
        else:
            await interaction.response.send_message("시간을 선택해주세요!", ephemeral=True, view=self)

    async def time_selected(self, interaction: discord.Interaction):
        self.selected_time = self.time_select.values[0]
        if self.selected_days:
            await self.save_reminder(interaction)
        else:
            await interaction.response.send_message("요일을 먼저 선택해주세요!", ephemeral=True, view=self)

    async def save_reminder(self, interaction):
        uid = self.uid
        reminder = {"days": list(self.selected_days), "time": self.selected_time}
        # 기존 reminder가 리스트가 아니면 변환
        if not isinstance(user_profiles[uid].get("reminder"), list):
            old = user_profiles[uid].get("reminder")
            if isinstance(old, str):
                try:
                    days_part, time_part = old.rsplit(" ", 1)
                    days = [d.strip() for d in days_part.split(",") if d.strip()]
                    user_profiles[uid]["reminder"] = [{"days": days, "time": time_part}]
                except:
                    user_profiles[uid]["reminder"] = []
            else:
                user_profiles[uid]["reminder"] = []
        # 중복 방지: 동일한 days+time이 이미 있으면 추가하지 않음
        exists = any(r["days"] == reminder["days"] and r["time"] == reminder["time"] for r in user_profiles[uid]["reminder"])
        if exists:
            await interaction.response.send_message(f"이미 동일한 리마인더가 등록되어 있습니다! (요일: {', '.join(self.selected_days)}, 시간: {self.selected_time})", ephemeral=True)
            return
        user_profiles[uid]["reminder"].append(reminder)
        save_user_profiles()
        await interaction.response.send_message(
            f"✅ 리마인더가 추가되었습니다! (요일: {', '.join(self.selected_days)}, 시간: {self.selected_time})",
            ephemeral=True,
            view=ReminderAddView(self.uid)  # 버튼 갱신
        )

@bot.tree.command(name="리마인더추가", description="리마인더 추가 (여러 요일/시간 지원)")
async def 리마인더추가(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
    else:
        await interaction.response.send_message("리마인더 요일과 시간을 선택하세요:", view=ReminderAddView(uid), ephemeral=True)

# === /리마인더 목록 명령어 ===
@bot.tree.command(name="리마인더목록", description="내 리마인더 전체 목록 보기")
async def 리마인더목록(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
        return
    reminders = user_profiles[uid].get("reminder", [])
    if not reminders:
        await interaction.response.send_message("등록된 리마인더가 없습니다.", ephemeral=True)
        return
    msg = "**[내 리마인더 목록]**\n"
    for idx, r in enumerate(reminders):
        msg += f"{idx+1}. 요일: {', '.join(r['days'])}, 시간: {r['time']}\n"
    await interaction.response.send_message(msg, ephemeral=True)

# === /리마인더 해제 명령어 ===
@bot.tree.command(name="리마인더해제", description="리마인더 삭제 (번호로 선택)")
@app_commands.describe(번호="/리마인더목록에서 확인한 리마인더 번호")
async def 리마인더해제(interaction: discord.Interaction, 번호: int):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
        return
    reminders = user_profiles[uid].get("reminder", [])
    if not reminders or 번호 < 1 or 번호 > len(reminders):
        await interaction.response.send_message("잘못된 번호입니다.", ephemeral=True)
        return
    removed = reminders.pop(번호-1)
    save_user_profiles()
    await interaction.response.send_message(f"✅ 리마인더가 삭제되었습니다. (요일: {', '.join(removed['days'])}, 시간: {removed['time']})", ephemeral=True)

# === /초기화 명령어 ===
class ResetConfirmModal(Modal):
    def __init__(self, uid):
        super().__init__(title="초기화 확인")
        self.uid = uid
        self.nickname = TextInput(label="닉네임을 입력하세요 (확인용)", placeholder="본인 닉네임 입력")
        self.add_item(self.nickname)

    async def on_submit(self, interaction):
        global user_profiles, user_dataset
        uid = self.uid
        if uid not in user_profiles:
            await interaction.response.send_message("❌ 등록된 정보가 없습니다.", ephemeral=True)
            return
        input_nick = self.nickname.value.strip()
        real_nick = user_profiles[uid]["nickname"].strip()
        if input_nick != real_nick:
            await interaction.response.send_message("❌ 닉네임이 일치하지 않습니다. 다시 시도해주세요.", ephemeral=True)
            return
        # 프로필에서 본인 정보만 삭제
        del user_profiles[uid]
        save_user_profiles()  # JSON으로 저장
        # 입력 데이터에서 본인 데이터만 삭제
        if not user_dataset.empty and "uid" in user_dataset.columns:
            user_dataset = user_dataset[user_dataset["uid"].astype(str) != uid]
            user_dataset.to_csv(user_dataset_file, index=False)
        await interaction.response.send_message("✅ 내 정보와 입력 데이터가 모두 초기화되었습니다.", ephemeral=True)

@bot.tree.command(name="초기화", description="내 정보와 입력 데이터 초기화")
async def 초기화(interaction: discord.Interaction):
    uid = str(interaction.user.id)
    if uid not in user_profiles:
        await interaction.response.send_message("먼저 `/생성` 명령어로 등록해주세요.", ephemeral=True)
    else:
        await interaction.response.send_modal(ResetConfirmModal(uid))
    
    # 기존 reminder 문자열을 리스트로 변환
    for uid, profile in user_profiles.items():
        reminder = profile.get("reminder")
        if isinstance(reminder, str):
            # 예: "월, 화, 수, 목, 금, 토, 일 21:00"
            try:
                days_part, time_part = reminder.rsplit(" ", 1)
                days = [d.strip() for d in days_part.split(",") if d.strip()]
                user_profiles[uid]["reminder"] = [{"days": days, "time": time_part}]
            except Exception as e:
                user_profiles[uid]["reminder"] = []
    save_user_profiles()

# === 관리자 명령어 ===

# 여기에 "봇 관리자"로 허용할 Discord 사용자 ID 추가
BOT_ADMINS = [464425198808989697]  # ← 본인 Discord ID로 바꿔주세요

class StatusChangeView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.status = None
        self.activity_type = None
        self.activity_name = None

    @discord.ui.select(
        placeholder="봇의 상태를 선택하세요",
        options=[
            discord.SelectOption(label="온라인", value="online", emoji="🟢"),
            discord.SelectOption(label="자리비움", value="idle", emoji="🌙"),
            discord.SelectOption(label="방해금지", value="dnd", emoji="⛔"),
            discord.SelectOption(label="오프라인(숨김)", value="invisible", emoji="⚫"),
        ]
    )
    async def select_status(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.status = select.values[0]
        await interaction.response.send_message(f"✅ 상태 선택됨: `{select.values[0]}`", ephemeral=True)

    @discord.ui.select(
        placeholder="활동 유형을 선택하세요",
        options=[
            discord.SelectOption(label="플레이 중", value="playing", emoji="🎮"),
            discord.SelectOption(label="듣는 중", value="listening", emoji="🎧"),
            discord.SelectOption(label="보는 중", value="watching", emoji="👀"),
            discord.SelectOption(label="참가 중", value="competing", emoji="🏆"),
        ]
    )
    async def select_activity(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.activity_type = select.values[0]
        await interaction.response.send_message(f"✅ 활동 유형 선택됨: `{select.values[0]}`", ephemeral=True)

    @discord.ui.button(label="활동 이름 입력 및 최종 적용", style=discord.ButtonStyle.green)
    async def input_activity(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(StatusModal(self))


class StatusModal(discord.ui.Modal, title="활동 이름 입력"):
    activity_name = discord.ui.TextInput(label="활동 메시지를 입력하세요", placeholder="/도움 을 입력해보세요")

    def __init__(self, view: StatusChangeView):
        super().__init__()
        self.view_ref = view

    async def on_submit(self, interaction: discord.Interaction):
        self.view_ref.activity_name = self.activity_name.value

        status_map = {
            "online": discord.Status.online,
            "idle": discord.Status.idle,
            "dnd": discord.Status.dnd,
            "invisible": discord.Status.invisible
        }
        activity_map = {
            "playing": discord.Game,
            "listening": lambda name: discord.Activity(type=discord.ActivityType.listening, name=name),
            "watching": lambda name: discord.Activity(type=discord.ActivityType.watching, name=name),
            "competing": lambda name: discord.Activity(type=discord.ActivityType.competing, name=name),
        }

        status = status_map.get(self.view_ref.status, discord.Status.online)
        activity_type = self.view_ref.activity_type or "playing"
        name = self.view_ref.activity_name or "/도움 을 입력해보세요"
        activity = activity_map[activity_type](name)

        await bot.change_presence(status=status, activity=activity)
        await interaction.response.send_message(
            f"🤖 봇 상태가 변경되었습니다:\n"
            f"🟢 상태: `{self.view_ref.status}`\n"
            f"🎮 활동: `{activity_type} {name}`", ephemeral=True)


@bot.command()
async def 상태변경(ctx):
    if ctx.author.id not in BOT_ADMINS:
        await ctx.send("🚫 이 명령어는 **봇 관리자**만 사용할 수 있습니다.")
        return

    view = StatusChangeView()
    await ctx.send("📲 봇 상태를 설정하세요!", view=view)




if __name__ == "__main__":
    bot.run(token)
