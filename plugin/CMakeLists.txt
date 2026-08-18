# Cleer as a VST3 / AU plug-in.
#
#   cmake -B build -DCLEER_JUCE_PATH=/path/to/JUCE   # or let it fetch 8.0.15
#   cmake --build build
#   ctest --test-dir build                            # offline DSP harness
#
# The DSP core (Source/dsp/) is header-only and framework-free; the harness
# builds and runs it without JUCE, mirroring the repo rule that the DSP math
# stays provable outside any host.

cmake_minimum_required(VERSION 3.22)

project(CleerPlugin VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(CLEER_BUILD_PLUGIN "Build the VST3 / AU / Standalone plug-in (needs JUCE)" ON)
option(CLEER_BUILD_TESTS  "Build the offline DSP test harness" ON)
option(CLEER_COPY_PLUGIN  "Copy the built plug-in into the user plug-in folders" OFF)
set(CLEER_JUCE_PATH "" CACHE PATH "Path to an existing JUCE checkout (optional)")

# ---------------------------------------------------------------------------
# Offline harness — no JUCE required.
# ---------------------------------------------------------------------------
if (CLEER_BUILD_TESTS)
    add_executable(cleer_dsp_tests Tests/DspTests.cpp)
    target_compile_features(cleer_dsp_tests PRIVATE cxx_std_17)
    if (NOT MSVC)
        target_compile_options(cleer_dsp_tests PRIVATE -Wall -Wextra)
    endif()
    enable_testing()
    add_test(NAME cleer_dsp_tests COMMAND cleer_dsp_tests)
endif()

# ---------------------------------------------------------------------------
# The plug-in itself.
# ---------------------------------------------------------------------------
if (CLEER_BUILD_PLUGIN)
    if (CLEER_JUCE_PATH AND EXISTS "${CLEER_JUCE_PATH}/CMakeLists.txt")
        message(STATUS "Cleer: using JUCE from CLEER_JUCE_PATH (${CLEER_JUCE_PATH})")
        add_subdirectory(${CLEER_JUCE_PATH} ${CMAKE_BINARY_DIR}/JUCE EXCLUDE_FROM_ALL)
    elseif (DEFINED ENV{JUCE_PATH} AND EXISTS "$ENV{JUCE_PATH}/CMakeLists.txt")
        message(STATUS "Cleer: using JUCE from $ENV{JUCE_PATH}")
        add_subdirectory($ENV{JUCE_PATH} ${CMAKE_BINARY_DIR}/JUCE EXCLUDE_FROM_ALL)
    elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/CMakeLists.txt")
        message(STATUS "Cleer: using bundled JUCE checkout")
        add_subdirectory(JUCE EXCLUDE_FROM_ALL)
    else()
        message(STATUS "Cleer: fetching JUCE 8.0.15")
        include(FetchContent)
        FetchContent_Declare(JUCE
            GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
            GIT_TAG 8.0.15
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(JUCE)
    endif()

    set(CLEER_FORMATS VST3 Standalone)
    if (APPLE)
        list(APPEND CLEER_FORMATS AU)
    endif()

    juce_add_plugin(Cleer
        COMPANY_NAME              "trv"
        BUNDLE_ID                 "as.trv.cleer.plugin"
        PLUGIN_MANUFACTURER_CODE  Trva
        PLUGIN_CODE               Clr1
        FORMATS                   ${CLEER_FORMATS}
        PRODUCT_NAME              "Cleer"
        DESCRIPTION               "De-feedback for live vocals: adaptive howl notching + downward expander"
        IS_SYNTH                  FALSE
        NEEDS_MIDI_INPUT          FALSE
        NEEDS_MIDI_OUTPUT         FALSE
        IS_MIDI_EFFECT            FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        MICROPHONE_PERMISSION_ENABLED TRUE
        MICROPHONE_PERMISSION_TEXT    "Cleer processes live audio input (standalone mode)."
        COPY_PLUGIN_AFTER_BUILD   ${CLEER_COPY_PLUGIN}
        VST3_CATEGORIES           "Fx" "Restoration"
        AU_MAIN_TYPE              kAudioUnitType_Effect)

    target_sources(Cleer PRIVATE
        Source/PluginProcessor.cpp)

    target_compile_definitions(Cleer PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0)

    target_link_libraries(Cleer
        PRIVATE
            juce::juce_audio_utils
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
            juce::juce_recommended_warning_flags)

    # JUCE writes moduleinfo.json into the bundle after the binary is
    # linker-signed, which breaks the signature seal — Ableton's scanner then
    # rejects the plug-in with "a sealed resource is missing or invalid"
    # (auval happens to tolerate it). Re-sign ad hoc as the last post-build
    # step so the installed bundle always validates strictly.
    if (APPLE)
        foreach(fmt IN ITEMS VST3 AU Standalone)
            if (TARGET Cleer_${fmt})
                add_custom_command(TARGET Cleer_${fmt} POST_BUILD
                    COMMAND codesign --force --deep -s - "$<TARGET_BUNDLE_DIR:Cleer_${fmt}>"
                    COMMENT "Re-sealing Cleer_${fmt} signature (ad hoc)")
            endif()
        endforeach()
    endif()
endif()
