#!/bin/bash
# 模型下载脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 AutoIF 模型下载脚本${NC}"
echo "================================"

# 检查 modelscope 是否安装
if ! python -c "import modelscope" &> /dev/null; then
    echo -e "${YELLOW}⚠️  ModelScope 未安装，正在安装...${NC}"
    pip install modelscope -q
    echo -e "${GREEN}✅ ModelScope 安装完成${NC}"
fi

# 设置模型保存路径
MODEL_DIR=${MODEL_DIR:-"./models"}
mkdir -p $MODEL_DIR

# 教师模型选择
echo ""
echo -e "${YELLOW}请选择教师模型（用于数据生成）:${NC}"
echo "1) Qwen/Qwen2-72B-Instruct (推荐，72B)"
echo "2) Qwen/Qwen2.5-72B-Instruct (最新，72B)"
echo "3) Qwen/Qwen2-7B-Instruct (轻量，7B)"
echo "4) 跳过（已有教师模型或使用API）"
read -p "请输入选项 [1-4]: " teacher_choice

case $teacher_choice in
    1)
        TEACHER_MODEL="Qwen/Qwen2-72B-Instruct"
        ;;
    2)
        TEACHER_MODEL="Qwen/Qwen2.5-72B-Instruct"
        ;;
    3)
        TEACHER_MODEL="Qwen/Qwen2-7B-Instruct"
        ;;
    4)
        echo -e "${GREEN}✅ 跳过教师模型下载${NC}"
        TEACHER_MODEL=""
        ;;
    *)
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
        ;;
esac

if [ ! -z "$TEACHER_MODEL" ]; then
    echo -e "${GREEN}📥 下载教师模型: $TEACHER_MODEL${NC}"
    python -c "from modelscope import snapshot_download; snapshot_download('$TEACHER_MODEL', cache_dir='$MODEL_DIR/teacher')"
    echo -e "${GREEN}✅ 教师模型下载完成${NC}"
fi

# 学生模型选择
echo ""
echo -e "${YELLOW}请选择学生模型（基础模型B）:${NC}"
echo "1) Qwen/Qwen2-7B (推荐，7B)"
echo "2) Qwen/Qwen2-1.5B (轻量，1.5B)"
echo "3) Qwen/Qwen2-0.5B (超轻量，0.5B)"
echo "4) meta-llama/Llama-3.2-3B (Llama系列，3B)"
echo "5) 跳过（已有学生模型）"
read -p "请输入选项 [1-5]: " student_choice

case $student_choice in
    1)
        STUDENT_MODEL="Qwen/Qwen2-7B"
        ;;
    2)
        STUDENT_MODEL="Qwen/Qwen2-1.5B"
        ;;
    3)
        STUDENT_MODEL="Qwen/Qwen2-0.5B"
        ;;
    4)
        STUDENT_MODEL="meta-llama/Llama-3.2-3B"
        ;;
    5)
        echo -e "${GREEN}✅ 跳过学生模型下载${NC}"
        STUDENT_MODEL=""
        ;;
    *)
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
        ;;
esac

if [ ! -z "$STUDENT_MODEL" ]; then
    echo -e "${GREEN}📥 下载学生模型: $STUDENT_MODEL${NC}"
    python -c "from modelscope import snapshot_download; snapshot_download('$STUDENT_MODEL', cache_dir='$MODEL_DIR/student')"
    echo -e "${GREEN}✅ 学生模型下载完成${NC}"
fi

# 下载辅助模型
echo ""
echo -e "${YELLOW}是否下载辅助模型？${NC}"
echo "1) mDeBERTa-v3 (NLI一致性检查)"
echo "2) BGE Embedding (RAG检索)"
echo "3) 全部下载"
echo "4) 跳过"
read -p "请输入选项 [1-4]: " aux_choice

download_mdeberta() {
    echo -e "${GREEN}📥 下载 mDeBERTa-v3-base-xnli${NC}"
    python -c "from modelscope import snapshot_download; snapshot_download('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7', cache_dir='$MODEL_DIR/mdeberta')"
}

download_bge() {
    echo -e "${GREEN}📥 下载 BGE Embedding${NC}"
    python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir='$MODEL_DIR/bge')"
}

case $aux_choice in
    1)
        download_mdeberta
        ;;
    2)
        download_bge
        ;;
    3)
        download_mdeberta
        download_bge
        ;;
    4)
        echo -e "${GREEN}✅ 跳过辅助模型下载${NC}"
        ;;
    *)
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 所有模型下载完成！${NC}"
echo "================================"
echo "模型保存路径: $MODEL_DIR"
ls -lh $MODEL_DIR/
