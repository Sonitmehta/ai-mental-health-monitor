/**
 * MindScan AI — Global Frontend Script
 * Theme Toggle (Light/Dark), Multilingual Translations (EN, HI, HL), Toast alerts, Mobile Nav
 */

// ─── Theme Management ────────────────────────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem("mhm_theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(savedTheme);

  const toggleBtns = document.querySelectorAll("#themeToggle");
  toggleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("mhm_theme", next);
      updateThemeIcon(next);
      showToast(`Switched to ${next.toUpperCase()} mode`, "success");
    });
  });
}

function updateThemeIcon(theme) {
  const icons = document.querySelectorAll("#themeIcon");
  icons.forEach(ic => {
    ic.textContent = theme === "dark" ? "☀️" : "🌙";
  });
}

// ─── Multilingual Translations ───────────────────────────────────────────────
const I18N = {
  en: {
    brand: "MindScan AI",
    nav_home: "Home",
    nav_screening: "Daily Check-In",
    nav_history: "My History",
    nav_about: "AI Model & Viva",
    nav_logout: "Logout",
    home_tag: "Personal Wellness Hub",
    home_sub: "Take a quiet moment to reflect on your day. Our AI is here to help you identify emotional trends and nurture your peace of mind.",
    btn_daily_checkin: "Start Daily Check-in",
    tile_last_status: "Latest Emotional State",
    no_checkin_yet: "Not checked in",
    take_first_checkin: "Take your first check-in today",
    tile_total_checkins: "Total Screenings",
    logged_in_records: "Saved in your private history",
    tile_privacy: "Privacy & Confidentiality",
    privacy_desc: "Evaluated securely in real-time",
    tip_heading: "Mindfulness Thought for Today",
    daily_quote: '"You don\'t have to control your thoughts. You just have to stop letting them control you."',
    daily_subtip: "Tip: When feeling overwhelmed, pause and take three slow, deliberate belly breaths.",
    how_it_helps_badge: "Simple & Accessible",
    how_it_helps_title: "How MindScan Supports Your Mental Health",
    step1_title: "1. Express Freely",
    step1_desc: "Write your thoughts or journal your day in natural conversational language.",
    step2_title: "2. Immediate Insight",
    step2_desc: "Our neural network detects subtle stress biomarkers and emotional signals.",
    step3_title: "3. Actionable Coping",
    step3_desc: "Get tailored relaxation steps, breathing exercises, and verified helpline contacts.",
    link_view_ai_tech: "Looking for technical AI accuracy & deep learning architecture? View Model Blueprint →",
    tab_signin: "Sign In",
    tab_signup: "Create Account",
    lbl_user: "Username or Email",
    lbl_pwd: "Password",
    lbl_name: "Full Name",
    lbl_email: "Email Address",
    lbl_set_pwd: "Create Password",
    btn_signin: "Sign In",
    btn_signup: "Create My Account",
    demo_or: "Or test without creating an account:",
    quick_demo_btn: "Instant Demo Login (1-Click)",
    screening_badge: "Daily Self-Check",
    screening_title: "How are you feeling today?",
    screening_desc: "Write freely about your day, thoughts, feelings, or stressors. The AI evaluates emotional nuance and provides tailored coping recommendations.",
    journal_label: "Your Thoughts & Reflections:",
    sample_prompts_label: "💡 Quick Test Prompts:",
    clear_btn: "Clear",
    analyze_btn: "Evaluate My Check-In",
    analysis_results_title: "Evaluation Summary",
    risk_score_label: "Overall Distress Index",
    detected_emotion_label: "Identified Emotional Tone",
    nlp_markers_label: "Distress Markers:",
    verified_helplines_label: "📞 24/7 Verified Confidential Helplines",
    prob_dist_label: "Neural Network Confidence Probabilities",
    disclaimer_note: "⚠️ Educational & awareness tool. Not a clinical replacement for professional diagnosis.",
    view_in_history_btn: "View in History & Trends →",
    history_badge: "Longitudinal Analytics",
    history_title: "Session History & Trends",
    history_desc: "Track mental wellbeing variations and risk score trajectories across screening interactions.",
    export_csv: "Export CSV",
    clear_history: "Clear History",
    stat_total_sessions: "Total Screenings",
    stat_avg_score: "Average Risk Score",
    stat_peak_score: "Peak Risk Recorded",
    trend_chart_badge: "Telemetry Curve",
    trend_chart_title: "Risk Score Progression",
    logs_badge: "Audit Log",
    logs_title: "Screening Entry Records",
    th_id: "ID",
    th_timestamp: "Timestamp",
    th_text: "Journal Excerpt",
    th_emotion: "Emotion",
    th_risk: "Risk Level",
    th_score: "Score"
  },
  hi: {
    brand: "माइंडस्कैन एआई",
    nav_home: "होम",
    nav_screening: "दैनिक जांच",
    nav_history: "मेरा इतिहास",
    nav_about: "एआई मॉडल एवं वाइवा",
    nav_logout: "लॉग आउट",
    home_tag: "व्यक्तिगत स्वास्थ्य केंद्र",
    home_sub: "अपने दिन पर विचार करने के लिए एक शांत क्षण निकालें। हमारा AI आपकी भावनाओं को समझने और मानसिक शांति बनाए रखने में मदद करता है।",
    btn_daily_checkin: "दैनिक जांच शुरू करें",
    tile_last_status: "नवीनतम भावनात्मक स्थिति",
    no_checkin_yet: "अभी जांच नहीं हुई",
    take_first_checkin: "आज अपनी पहली जांच करें",
    tile_total_checkins: "कुल जांच सत्र",
    logged_in_records: "आपके निजी इतिहास में सुरक्षित",
    tile_privacy: "गोपनीयता और सुरक्षा",
    privacy_desc: "वास्तविक समय में सुरक्षित मूल्यांकन",
    tip_heading: "आज का माइंडफुलनेस विचार",
    daily_quote: '"आपको अपने विचारों को नियंत्रित करने की आवश्यकता नहीं है। बस उन्हें खुद पर नियंत्रण न करने दें।"',
    daily_subtip: "सुझाव: जब तनाव महसूस हो, तो रुकें और तीन गहरी सांसें लें।",
    how_it_helps_badge: "सरल एवं सुलभ",
    how_it_helps_title: "माइंडस्कैन आपके स्वास्थ्य का समर्थन कैसे करता है",
    step1_title: "1. खुलकर लिखें",
    step1_desc: "अपनी स्वाभाविक भाषा में अपने विचार या दिन की बातें लिखें।",
    step2_title: "2. त्वरित जानकारी",
    step2_desc: "हमारा न्यूरल नेटवर्क तनाव के संकेतों और भावनाओं को पहचानता है।",
    step3_title: "3. व्यावहारिक उपाय",
    step3_desc: "तनावमुक्ति के चरण, श्वास अभ्यास और सत्यापित हेल्पलाइन नंबर प्राप्त करें।",
    link_view_ai_tech: "तकनीकी विवरण और डीप लर्निंग मॉडल देखना चाहते हैं? मॉडल ब्लूप्रिंट देखें →",
    tab_signin: "साइन इन",
    tab_signup: "खाता बनाएं",
    lbl_user: "यूज़रनेम या ईमेल",
    lbl_pwd: "पासवर्ड",
    lbl_name: "पूरा नाम",
    lbl_email: "ईमेल पता",
    lbl_set_pwd: "पासवर्ड बनाएं",
    btn_signin: "साइन इन करें",
    btn_signup: "खाता बनाएं",
    demo_or: "या बिना खाता बनाए तुरंत टेस्ट करें:",
    quick_demo_btn: "1-क्लिक डेमो लॉगिन",
    screening_badge: "दैनिक स्व-मूल्यांकन",
    screening_title: "आज आप कैसा महसूस कर रहे हैं?",
    screening_desc: "अपने विचारों या तनाव के बारे में खुलकर लिखें। AI आपकी भावनाओं का विश्लेषण करके सही सलाह प्रदान करता है।",
    journal_label: "आपके विचार / जर्नल:",
    sample_prompts_label: "💡 त्वरित परीक्षण संकेत:",
    clear_btn: "साफ़ करें",
    analyze_btn: "मूल्यांकन करें",
    analysis_results_title: "मूल्यांकन सारांश",
    risk_score_label: "कुल तनाव सूचकांक",
    detected_emotion_label: "पहचानी गई भावना",
    nlp_markers_label: "तनाव के संकेत:",
    verified_helplines_label: "📞 24/7 सत्यापित गोपनीय हेल्पलाइन",
    prob_dist_label: "न्यूरल नेटवर्क आउटपुट संभावनाएं",
    disclaimer_note: "⚠️ यह केवल जागरूकता उपकरण है, डॉक्टरी निदान का विकल्प नहीं।",
    view_in_history_btn: "इतिहास में देखें →",
    history_badge: "दीर्घकालिक विश्लेषण",
    history_title: "सत्र इतिहास और रुझान",
    history_desc: "समय के साथ अपनी मानसिक स्थिति को ट्रैक करें।",
    export_csv: "सीएसवी निर्यात",
    clear_history: "इतिहास मिटाएं",
    stat_total_sessions: "कुल जांच",
    stat_avg_score: "औसत स्कोर",
    stat_peak_score: "उच्चतम स्कोर",
    trend_chart_badge: "प्रगति वक्र",
    trend_chart_title: "जोखिम स्कोर प्रगति",
    logs_badge: "ऑडिट लॉग",
    logs_title: "प्रविष्टियों का रिकॉर्ड",
    th_id: "आईडी",
    th_timestamp: "समय",
    th_text: "जर्नल टेक्स्ट",
    th_emotion: "भावना",
    th_risk: "जोखिम स्तर",
    th_score: "स्कोर"
  },
  hl: {
    brand: "MindScan AI",
    nav_home: "Home",
    nav_screening: "Daily Check-In",
    nav_history: "My History",
    nav_about: "AI Model & Viva",
    nav_logout: "Logout",
    home_tag: "Personal Wellness Hub",
    home_sub: "Apne din ke baare me sochein. Hamara AI emotional trends identify karke aapko sukoon dene me madad karta hai.",
    btn_daily_checkin: "Daily Check-in Shuru Karein",
    tile_last_status: "Last Emotional State",
    no_checkin_yet: "Abhi test nahi kiya",
    take_first_checkin: "Aaj pehla check-in karein",
    tile_total_checkins: "Total Screenings",
    logged_in_records: "Aapki private history me saved",
    tile_privacy: "100% Private & Safe",
    privacy_desc: "Real-time me secure evaluation",
    tip_heading: "Aaj Ka Mindfulness Thought",
    daily_quote: '"Apne har thought ko control karne ki zaroorat nahi hai. Bas unhe khud par haavi mat hone do."',
    daily_subtip: "Tip: Jab bohot zyada stress ho, toh 3 baar gehri saans lein.",
    how_it_helps_badge: "Simple & Easy",
    how_it_helps_title: "MindScan Kaise Help Karta Hai",
    step1_title: "1. Freely Likhein",
    step1_desc: "Apne thoughts aur din ki baatein naturally likhein.",
    step2_title: "2. Instant Insight",
    step2_desc: "Neural network stress keywords aur emotion detect karta hai.",
    step3_title: "3. Actionable Advice",
    step3_desc: "Breathing steps, relaxation tips aur official verified helpline numbers paayein.",
    link_view_ai_tech: "Technical AI metrics aur deep learning architecture dekhna hai? Blueprint kholein →",
    tab_signin: "Sign In",
    tab_signup: "Naya Account",
    lbl_user: "Username ya Email",
    lbl_pwd: "Password",
    lbl_name: "Pura Naam",
    lbl_email: "Email Address",
    lbl_set_pwd: "Naya Password",
    btn_signin: "Sign In Karein",
    btn_signup: "Account Banayein",
    demo_or: "Ya bina account banaye direct test karein:",
    quick_demo_btn: "1-Click Demo Login",
    screening_badge: "Daily Self-Check",
    screening_title: "Aaj kaisa feel kar rahe hain?",
    screening_desc: "Apne thoughts ya feelings likhein. AI real-time me stress aur emotion check karke relaxation steps batayega.",
    journal_label: "Apne thoughts yahan likhein:",
    sample_prompts_label: "💡 Test karne ke liye sample prompt chunein:",
    clear_btn: "Clear",
    analyze_btn: "Check-in Evaluate Karein",
    analysis_results_title: "Evaluation Summary",
    risk_score_label: "Total Distress Score",
    detected_emotion_label: "Detected Emotion",
    nlp_markers_label: "Stress Markers:",
    verified_helplines_label: "📞 24/7 Verified Free Helplines",
    prob_dist_label: "Neural Network Probabilities",
    disclaimer_note: "⚠️ Yeh educational tool hai, doctor ki advice ka replacement nahi.",
    view_in_history_btn: "History me dekhein →",
    history_badge: "Progress Tracking",
    history_title: "Past Check-ins & Graph",
    history_desc: "Apne purane check-in records aur score chart dekhein.",
    export_csv: "CSV Download",
    clear_history: "History Clear Karo",
    stat_total_sessions: "Total Tests",
    stat_avg_score: "Average Score",
    stat_peak_score: "Highest Score",
    trend_chart_badge: "Score Progression",
    trend_chart_title: "Score Over Time",
    logs_badge: "Logs",
    logs_title: "Saved Session Entries",
    th_id: "ID",
    th_timestamp: "Time",
    th_text: "Journal Line",
    th_emotion: "Emotion",
    th_risk: "Risk",
    th_score: "Score"
  }
};

function applyTranslations(lang) {
  const dict = I18N[lang] || I18N.en;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (dict[key]) {
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
        el.placeholder = dict[key];
      } else {
        el.textContent = dict[key];
      }
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();

  const savedLang = localStorage.getItem("mhm_lang") || "en";
  const buttons = document.querySelectorAll(".lang-btn");
  buttons.forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-lang") === savedLang);
    btn.addEventListener("click", () => {
      const selected = btn.getAttribute("data-lang");
      localStorage.setItem("mhm_lang", selected);
      buttons.forEach(b => b.classList.toggle("active", b.getAttribute("data-lang") === selected));
      applyTranslations(selected);
      showToast(`Language: ${selected.toUpperCase()}`, "success");
    });
  });

  applyTranslations(savedLang);

  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks") || document.getElementById("nav-links");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });
  }
});

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
