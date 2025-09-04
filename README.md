# Форк llama.cpp (работа, в основном инференс, с LLM в формате .gguf на C/C++) чтобы создать на его основе инструмент обучения нейросетей на ограниченном оборудовании
[ReadMe оригинала](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Изменения в llama.cpp с 2024 года
[Документация по работе с новыми весиями llama.cpp](https://dzen.ru/a/aK6m7LtcORy66-Po)

## Запуск инструмента анализа вероятности следующего токена:
* Скачать [релиз](https://github.com/no4ni/llama.cpp/releases)
* Разархивировать
* Запустить через cmd или Windows Terninal (для UTF-8)? например:
  ```
  prediction-next-token -m "models\llama-3.2-1B-q4_k_m-128k.gguf" -p "Who invented E=mc^2?" -h "Einstein"
  ```

## Сборка для разработчиков:
1.
```
git clone https://github.com/no4ni/llama.cpp
cd llama.cpp
```
2. Для VS2022 на Windows: (для остального инструкции [здесь](https://github.com/no4ni/llama.cpp/blob/main/docs/build.md))
```
mkdir build
cd build
```
* Для простого проекта на СPU
```
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON
```
* Для поддержки GPU
```
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON
```
В **llama.cpp/build** появится **llama.cpp.sln**

## Тестовая русская нейросеть
Можно скачать [отсюда](https://huggingface.co/Vikhrmodels/Vikhr-Llama-3.2-1B-instruct-GGUF) 
(Для VRAM 4GB и RAM 8GB рекомендуется Vikhr-Llama-3.2-1B-Q4_K_M.gguf 808 МБ и ставить -c 65536 вместо максимума)
