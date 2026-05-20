#!/bin/bash
# 模型下载脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$PROJECT_DIR/models"}
echo -e "${GREEN}🚀 AutoIF 模型下载脚本${NC}"
echo "================================"

if ! python -c "import modelscope" &> /dev/null; then
    echo -e "${YELLOW}⚠️  ModelScope 未安装，正在安装...${NC}"
    pip install modelscope -q
    echo -e "${GREEN}✅ ModelScope 安装完成${NC}"
fi

MODEL_DIR=${MODEL_DIR:-"../models"}
mkdir -p $MODEL_DIR

echo ""
echo -e "${YELLOW}关于教师模型:${NC}"
echo -e "当前配置为使用 DeepSeek API，因此跳过本地教师模型下载。"
TEACHER_MODEL=""

# 学生模型选择
echo ""
echo -e "${YELLOW}请选择学生模型:${NC}"
echo "1) Qwen/Qwen2.5-1.5B-Instruct (当前流程使用此版本)"
echo "2) Qwen/Qwen2.5-7B-Instruct (进阶，7B)"
echo "3) 跳过（已有学生模型）"
read -p "请输入选项 [1-3]: " student_choice

case $student_choice in
    1) STUDENT_MODEL="Qwen/Qwen2.5-1.5B-Instruct" ;;
    2) STUDENT_MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
    3)
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
echo "2) 跳过"
read -p "请输入选项 [1-2]: " aux_choice

case $aux_choice in
    1)
        echo -e "${GREEN}📥 下载 mDeBERTa-v3-base-xnli${NC}"
        python -c "from huggingface_hub import snapshot_download; snapshot_download('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7', local_dir='$MODEL_DIR/nli')"
        ;;
    2)
        echo -e "${GREEN}✅ 跳过辅助模型下载${NC}"
        ;;
    *)
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 模型处理完成！${NC}"
echo "================================"