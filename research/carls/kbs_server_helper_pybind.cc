/*Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "research/carls/kbs_server_helper.h"
#include "pybind11/pybind11.h"

namespace carls {

PYBIND11_MODULE(kbs_server_helper_pybind, m) {
  m.doc() = R"pbdoc(
    _pywrap_kbs_server_helper
    A module that returns KBS server helper
  )pbdoc";

  pybind11::class_<KnowledgeBankServiceOptions>(m,
                                                "KnowledgeBankServiceOptions")
      .def(pybind11::init<bool, int, int>())
      .def_readwrite("run_locally", &KnowledgeBankServiceOptions::run_locally)
      .def_readwrite("port", &KnowledgeBankServiceOptions::port)
      .def_readwrite("num_threads", &KnowledgeBankServiceOptions::num_threads);

  pybind11::class_<KbsServerHelper>(m, "KbsServerHelper")
      .def(pybind11::init<const KnowledgeBankServiceOptions&>())
      .def("WaitForTermination", &KbsServerHelper::WaitForTermination)
      .def("Terminate", &KbsServerHelper::Terminate)
      .def("address", &KbsServerHelper::address)
      .def("port", &KbsServerHelper::port);
}

}  // namespace carls
