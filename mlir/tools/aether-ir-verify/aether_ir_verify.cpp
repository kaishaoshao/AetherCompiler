//===-- aether_ir_verify.cpp - Standalone C++17 Aether MLIR Structural Verifier -*- C++ -*-===//
//
// Zero-LLVM-dependency C++17 verifier compiled via CMake / g++ to validate multi-dialect
// `.mlir` files exported by `compiler/backend/mlir_exporter.py` (`aether_graph`,
// `aether_mem`, `aether_place`):
//   1. Balanced module / func braces.
//   2. Valid dialect op prefixes (`aether_graph.*`, `aether_mem.*`, `aether_place.*`).
//   3. Strict SSA single-definition uniqueness across `%name` results.
//   4. Valid `#aether_mem.space<"tier">` and `#aether_place.spec<chip = "...", tp = N>` syntax.
//===--------------------------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: aether-ir-verify <file.mlir>\n";
    return 2;
  }

  std::ifstream in(argv[1]);
  if (!in.is_open()) {
    std::cerr << "[ERROR] Cannot open MLIR file: " << argv[1] << "\n";
    return 2;
  }

  std::string line;
  int braceDepth = 0;
  int lineNo = 0;
  int graphOps = 0, memOps = 0, placeOps = 0;
  bool inFuncHeader = false;
  std::set<std::string> definedSSA;
  std::vector<std::string> errors;

  std::regex argRe(R"(%([A-Za-z0-9_]+)\s*:)");
  std::regex opRe(R"(^\s*(%[^=]+)=\s*(aether_graph|aether_mem|aether_place)\.([a-z0-9_]+)\(([^)]*)\))");
  std::regex memSpaceRe(R"(#aether_mem\.space<\"([^\"]+)\">)");
  std::regex placeSpecRe(R"(#aether_place\.spec<chip\s*=\s*\"([^\"]+)\",\s*tp\s*=\s*([0-9]+)>)");

  while (std::getline(in, line)) {
    ++lineNo;
    for (char c : line) {
      if (c == '{') ++braceDepth;
      else if (c == '}') --braceDepth;
      if (braceDepth < 0)
        errors.push_back("Line " + std::to_string(lineNo) + ": unmatched '}'");
    }

    // Collect single-line or multi-line func.func arguments into SSA scope
    if (line.find("func.func") != std::string::npos) {
      inFuncHeader = true;
    }
    if (inFuncHeader) {
      auto begin = std::sregex_iterator(line.begin(), line.end(), argRe);
      auto end = std::sregex_iterator();
      for (auto it = begin; it != end; ++it) {
        definedSSA.insert((*it)[1].str());
      }
      if (line.find('{') != std::string::npos) {
        inFuncHeader = false;
      }
      continue;
    }

    std::smatch m;
    if (std::regex_search(line, m, opRe)) {
      std::string lhs = m[1].str();
      std::string dialect = m[2].str();
      std::string operands = m[4].str();

      if (dialect == "aether_graph") ++graphOps;
      else if (dialect == "aether_mem") ++memOps;
      else if (dialect == "aether_place") ++placeOps;

      // Verify all operand SSA values were defined prior to use
      std::regex valRe(R"(%([A-Za-z0-9_]+))");
      auto ob = std::sregex_iterator(operands.begin(), operands.end(), valRe);
      auto oe = std::sregex_iterator();
      for (auto it = ob; it != oe; ++it) {
        std::string v = (*it)[1].str();
        if (definedSSA.find(v) == definedSSA.end()) {
          errors.push_back("Line " + std::to_string(lineNo) + ": undefined SSA operand %" + v);
        }
      }

      // Register result SSA values & enforce single definition
      auto rb = std::sregex_iterator(lhs.begin(), lhs.end(), valRe);
      auto re = std::sregex_iterator();
      for (auto it = rb; it != re; ++it) {
        std::string r = (*it)[1].str();
        if (!definedSSA.insert(r).second) {
          errors.push_back("Line " + std::to_string(lineNo) + ": duplicate SSA definition %" + r);
        }
      }

      // Validate #aether_mem.space and #aether_place.spec encodings when present
      if (line.find("#aether_mem.space") != std::string::npos && !std::regex_search(line, memSpaceRe)) {
        errors.push_back("Line " + std::to_string(lineNo) + ": malformed #aether_mem.space attribute");
      }
      if (line.find("#aether_place.spec") != std::string::npos && !std::regex_search(line, placeSpecRe)) {
        errors.push_back("Line " + std::to_string(lineNo) + ": malformed #aether_place.spec attribute");
      }
    }
  }

  if (braceDepth != 0) {
    errors.push_back("Unbalanced module/func braces (depth=" + std::to_string(braceDepth) + ")");
  }

  if (!errors.empty()) {
    std::cerr << "[FAIL] Aether MLIR verification failed (" << errors.size() << " errors):\n";
    for (const auto &e : errors) std::cerr << "  - " << e << "\n";
    return 1;
  }

  std::cout << "[OK] Verified Aether MLIR (" << argv[1] << "): "
            << "aether_graph=" << graphOps << ", aether_mem=" << memOps
            << ", aether_place=" << placeOps << ", ssa_values=" << definedSSA.size() << "\n";
  return 0;
}
