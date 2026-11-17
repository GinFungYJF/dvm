/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DVM_EXPORT_H_
#define DVM_EXPORT_H_

/*
 * The library is built with -fvisibility=hidden, so every entry point that a
 * consumer links against has to be marked here. Anything left unmarked stays
 * internal to libdvm.so and will fail to link downstream.
 */
#if defined(_WIN32)
#if defined(DVM_STATIC)
#define DVM_API
#elif defined(DVM_BUILDING_LIBRARY)
#define DVM_API __declspec(dllexport)
#else
#define DVM_API __declspec(dllimport)
#endif
#else
#define DVM_API __attribute__((visibility("default")))
#endif

#endif  // DVM_EXPORT_H_
