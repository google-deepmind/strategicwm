# Copyright 2025 The strategicwm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize a game tree using PyVis."""

import re

import bs4
try:
  from google.colab import output
except ImportError:
  output = None
from IPython import display as display_html
import ipywidgets as widgets
import networkx as nx
import pyvis

from strategicwm._src.se.construction import io
from strategicwm._src.se.visualization import annotate


class GameTreeVis:
  """Visualize a game tree using PyVis."""
  pyvis_keep_keys = frozenset([
      "id",
      "font_family",
      "label",
      "title",
      "shape",
      "color",
      "size",
      "iss_group",
      "paths",
  ])

  def __init__(
      self,
      game_tree: io.GameTreeDict,
      config: annotate.ColorConfig,
  ):
    self.config = config
    # Annotate the game tree with colors, styles, and tooltips
    self.annotator = annotate.GameTreeAnnotator(config)
    game_tree_nx = self.annotator.annotate_game_tree(game_tree)
    self.game_tree_nx = self.strip_unnecessary_fields(game_tree_nx)
    self.iss_groups_to_ids = self.get_iss_group_to_ids(game_tree_nx)
    # Create separate holders for node and path selections
    self._node_selection_holder = {"selection": None}
    self._path_selection_holder = {"selection": None}
    self._html_content = ""

    leaf_nodes = set([])
    for node in game_tree_nx.nodes:
      if "paths" in game_tree_nx.nodes[node]:
        leaf_nodes.update(game_tree_nx.nodes[node]["paths"])
    self.leaf_nodes = sorted(leaf_nodes)

  def strip_unnecessary_fields(self, game_tree_nx: nx.DiGraph) -> nx.DiGraph:
    """Strips unnecessary fields from the game tree for lighter-weight html."""
    game_tree_nx_stripped = game_tree_nx.copy()
    for node in game_tree_nx.nodes:
      keys = list(game_tree_nx.nodes[node].keys())
      for key in keys:
        if key not in self.pyvis_keep_keys:
          del game_tree_nx_stripped.nodes[node][key]
    return game_tree_nx_stripped

  @property
  def node_selection(self):
    return self._node_selection_holder["selection"]

  @property
  def path_selection(self):
    return self._path_selection_holder["selection"]

  def set_node_selection(self, node_id: str):
    """Receives the selected node ID from JavaScript."""
    self._node_selection_holder["selection"] = node_id
    self._path_selection_holder["selection"] = None  # Clear the other selection
    return "OK"

  def set_path_selection(self, leaf_id: str):
    """Receives the selected leaf ID (representing a path) from JavaScript."""
    self._path_selection_holder["selection"] = leaf_id
    self._node_selection_holder["selection"] = None  # Clear the other selection
    return "OK"

  def get_iss_group_to_ids(
      self, game_tree_nx: nx.DiGraph
  ) -> dict[str, list[str]]:
    iss_groups = {}
    for node_id in game_tree_nx.nodes:
      v = game_tree_nx.nodes[node_id]
      if "iss_group" in v:
        iss_groups.setdefault(v["iss_group"], []).append(node_id)
    return iss_groups

  def html_from_nx(self) -> str:
    """Returns an HTML string from a game tree in NetworkX format."""

    net = pyvis.network.Network(
        notebook=True,
        height="750px",
        width="100%",
        cdn_resources="remote",
        directed=True,
        select_menu=True,
        filter_menu=False
    )
    net.from_nx(self.game_tree_nx)

    # Keep only essential options in set_options
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD"
        }
      },
      "physics": {
        "enabled": true,
        "solver": "barnesHut"
      }
    }
    """)

    # Save and display
    html_content = net.generate_html(notebook=True)
    soup = bs4.BeautifulSoup(html_content, "html.parser")

    # Register BOTH functions with unique names
    if output:
      output.register_callback("set_node_selection", self.set_node_selection)
      output.register_callback("set_path_selection", self.set_path_selection)

    js_modifications = """
    var node_select_ts = null;

    function sendNodeSelectionToPython(value) {
        google.colab.kernel.invokeFunction('set_node_selection', [value], {});
    }
    function sendPathSelectionToPython(value) {
        google.colab.kernel.invokeFunction('set_path_selection', [value], {});
    }

    function masterNodeSelectHandler(params) {
      // 1. Visually highlight the graph
      neighbourhoodHighlight(params);

      // 2. Update the TomSelect dropdown
      if (node_select_ts) {
        var nodeId = (params.nodes && params.nodes.length > 0) ? params.nodes[0] : '';
        // The 'true' flag prevents this from re-triggering the onchange event
        node_select_ts.setValue(nodeId, true);
      }

      // 3. Send the selection back to Python
      var nodeIdToSend = (params.nodes && params.nodes.length > 0) ? params.nodes[0] : null;
      sendNodeSelectionToPython(nodeIdToSend);
    }

    function selectNode(nodes) {
      masterNodeSelectHandler({nodes: nodes});
      return nodes;
    }

    function selectPathByLeaf(selectedLeafId) {
        if (!selectedLeafId) {
            masterNodeSelectHandler({nodes: []});
            sendPathSelectionToPython(null);
            return;
        }
        allNodes = nodes.get({returnType: 'Object'});
        var nodesOnPath = [];
        for (let nodeId in allNodes) {
            allNodes[nodeId].color = 'rgba(200,200,200,0.5)';
            if (allNodes[nodeId].hiddenLabel === undefined) {
                allNodes[nodeId].hiddenLabel = allNodes[nodeId].label;
                allNodes[nodeId].label = undefined;
            }

            if (allNodes[nodeId].paths && allNodes[nodeId].paths.includes(selectedLeafId)) {
                nodesOnPath.push(nodeId);
            }
        }
        for (var i = 0; i < nodesOnPath.length; i++) {
            var nodeId = nodesOnPath[i];
            if (allNodes[nodeId]) {
                allNodes[nodeId].color = nodeColors[nodeId];
                if (allNodes[nodeId].hiddenLabel !== undefined) {
                    allNodes[nodeId].label = allNodes[nodeId].hiddenLabel;
                    allNodes[nodeId].hiddenLabel = undefined;
                }
            }
        }
        network.selectNodes(nodesOnPath);
        var updateArray = [];
        for (let nodeId in allNodes) {
            if (allNodes.hasOwnProperty(nodeId)) {
                updateArray.push(allNodes[nodeId]);
            }
        }
        nodes.update(updateArray);
        highlightActive = true;
        sendPathSelectionToPython(selectedLeafId);
    }

    function selectInfoSet(selectedInfoSetId) {
        if (!selectedInfoSetId) {
            masterNodeSelectHandler({nodes: []});
            return;
        }
        allNodes = nodes.get({returnType: 'Object'});
        var nodesInInfoSet = [];
        for (let nodeId in allNodes) {
            allNodes[nodeId].color = 'rgba(200,200,200,0.5)';
            if (allNodes[nodeId].hiddenLabel === undefined) {
                allNodes[nodeId].hiddenLabel = allNodes[nodeId].label;
                allNodes[nodeId].label = undefined;
            }

            if (allNodes[nodeId].iss_group && allNodes[nodeId].iss_group == selectedInfoSetId) {
                nodesInInfoSet.push(nodeId);
            }
        }
        for (var i = 0; i < nodesInInfoSet.length; i++) {
            var nodeId = nodesInInfoSet[i];
            if (allNodes[nodeId]) {
                allNodes[nodeId].color = nodeColors[nodeId];
                if (allNodes[nodeId].hiddenLabel !== undefined) {
                    allNodes[nodeId].label = allNodes[nodeId].hiddenLabel;
                    allNodes[nodeId].hiddenLabel = undefined;
                }
            }
        }
        network.selectNodes(nodesInInfoSet);
        var updateArray = [];
        for (let nodeId in allNodes) {
            if (allNodes.hasOwnProperty(nodeId)) {
                updateArray.push(allNodes[nodeId]);
            }
        }
        nodes.update(updateArray);
        highlightActive = true;
    }

    function neighbourhoodHighlight(params) {
        allNodes = nodes.get({returnType: 'Object'});
        if (params.nodes.length > 0) {
            highlightActive = true;
            var selectedNode = params.nodes[0];
            // mark all nodes as hard to read...
            for (let nodeId in allNodes) {
              allNodes[nodeId].color = 'rgba(200,200,200,0.5)';
              if (allNodes[nodeId].hiddenLabel === undefined) { allNodes[nodeId].hiddenLabel = allNodes[nodeId].label; allNodes[nodeId].label = undefined; }
            }
            // highlight logic for 1st and 2nd degree nodes...
            var connectedNodes = network.getConnectedNodes(selectedNode);
            var allConnectedNodes = [];
            for (var i = 1; i < 2; i++) { for (var j = 0; j < connectedNodes.length; j++) { allConnectedNodes = allConnectedNodes.concat(network.getConnectedNodes(connectedNodes[j])); } }
            for (var i = 0; i < allConnectedNodes.length; i++) {
              allNodes[allConnectedNodes[i]].color = 'rgba(150,150,150,0.75)';
              if (allNodes[allConnectedNodes[i]].hiddenLabel !== undefined) { allNodes[allConnectedNodes[i]].label = allNodes[allConnectedNodes[i]].hiddenLabel; allNodes[allConnectedNodes[i]].hiddenLabel = undefined; }
            }
            for (var i = 0; i < connectedNodes.length; i++) {
              allNodes[connectedNodes[i]].color = nodeColors[connectedNodes[i]];
              if (allNodes[connectedNodes[i]].hiddenLabel !== undefined) { allNodes[connectedNodes[i]].label = allNodes[connectedNodes[i]].hiddenLabel; allNodes[connectedNodes[i]].hiddenLabel = undefined; }
            }
            allNodes[selectedNode].color = nodeColors[selectedNode];
            if (allNodes[selectedNode].hiddenLabel !== undefined) { allNodes[selectedNode].label = allNodes[selectedNode].hiddenLabel; allNodes[selectedNode].hiddenLabel = undefined; }
        } else if (highlightActive === true) {
            // reset all nodes...
            for (let nodeId in allNodes) {
              allNodes[nodeId].color = nodeColors[nodeId];
              if (allNodes[nodeId].hiddenLabel !== undefined) { allNodes[nodeId].label = allNodes[nodeId].hiddenLabel; allNodes[nodeId].hiddenLabel = undefined; }
            }
            highlightActive = false;
        }
        var updateArray = [];
        for (let nodeId in allNodes) { if (allNodes.hasOwnProperty(nodeId)) { updateArray.push(allNodes[nodeId]); } }
        nodes.update(updateArray);
    }
    """

    head_script = soup.head
    head_script_tag = None
    if head_script is not None:
      head_script_tag = head_script.find("script")
    if head_script_tag:
      head_script_tag.string = js_modifications

    body_script = soup.body
    body_script_tag = None
    if body_script is not None:
      body_script_tag = body_script.find("script", type="text/javascript")
    if body_script_tag and body_script_tag.string:

      # 1. Make `nodeColors` global by moving its initialization
      # Remove it from inside drawGraph()
      body_script_tag.string = body_script_tag.string.replace(
          "nodeColors = {};", "", 1
      )
      # Add it to the global scope at the top of the script
      body_script_tag.string = body_script_tag.string.replace(
          "var filter = {", "nodeColors = {};\n              var filter = {", 1
      )

      # 2. Store the TomSelect instance for the NODE selector
      body_script_tag.string = body_script_tag.string.replace(
          'new TomSelect("#select-node"',
          'node_select_ts = new TomSelect("#select-node"',
          1
      )

      # 3. Point the network's event listener to our master handler
      body_script_tag.string = body_script_tag.string.replace(
          'network.on("selectNode", neighbourhoodHighlight);',
          'network.on("selectNode", masterNodeSelectHandler);',
          1
      )

      # 4. Add the TomSelect initializer for the PATH selector
      # pylint: disable=f-string-without-interpolation
      tomselect_init_for_path = f"""
          new TomSelect("#select-path-by-leaf",{{
              create: false,
              allowEmptyOption: true,
              sortField: {{
                  field: "text",
                  direction: "asc"
              }}
          }});
      """
      tomselect_init_for_iss = f"""
          new TomSelect("#select-infoset-by-id",{{
              create: false,
              allowEmptyOption: true,
              sortField: {{
                  field: "text",
                  direction: "asc"
              }}
          }});
      """
      # pylint: enable=f-string-without-interpolation

      # Find the full block for the node selector that we just modified
      node_ts_block_search = re.search(
          r'(node_select_ts = new TomSelect\("#select-node".*?}\);)',
          body_script_tag.string,
          re.DOTALL)
      if node_ts_block_search:
        full_block = node_ts_block_search.group(1)
        # Append the new initializer right after it to avoid any conflicts
        body_script_tag.string = body_script_tag.string.replace(
            full_block,
            full_block
            + "\n"
            + tomselect_init_for_path
            + "\n"
            + tomselect_init_for_iss,
            1,
        )

    reset_button = soup.find(
        "button", onclick="neighbourhoodHighlight({nodes: []});")
    if reset_button:
      # Change the onclick to call our new selectNode function with an empty
      # array. This ensures the Python callback is triggered on reset.
      reset_button["onclick"] = "selectNode([]);"

    # Create and inject the HTML dropdown
    select_menu_div = soup.find("div", id="select-menu")
    if select_menu_div:
      path_row = soup.new_tag("div", **{"class": "row no-gutters"})
      select_col = soup.new_tag("div", **{"class": "col-10 pb-2"})
      path_row.append(select_col)
      path_select = soup.new_tag(
          "select",
          id="select-path-by-leaf",
          **{
              "class": "form-select",
              "onchange": "selectPathByLeaf(this.value)",
          },
      )
      select_col.append(path_select)
      default_option = soup.new_tag("option", value="")
      default_option.string = "Select a Path by its Leaf Node"
      default_option["selected"] = ""
      path_select.append(default_option)
      for leaf_n in self.leaf_nodes:
        option = soup.new_tag("option", value=leaf_n)
        option.string = f"Path to leaf: {leaf_n}"
        path_select.append(option)
      node_selector_row = select_menu_div.find("div", class_="row no-gutters")  # pylint: disable=attribute-error
      if node_selector_row:
        node_selector_row.insert_after(path_row)

    select_menu_div = soup.find("div", id="select-menu")
    if select_menu_div:
      path_row = soup.new_tag("div", **{"class": "row no-gutters"})
      select_col = soup.new_tag("div", **{"class": "col-10 pb-2"})
      path_row.append(select_col)
      path_select = soup.new_tag(
          "select",
          id="select-infoset-by-id",
          **{
              "class": "form-select",
              "onchange": "selectInfoSet(this.value)",
          },
      )
      select_col.append(path_select)
      default_option = soup.new_tag("option", value="")
      default_option.string = "Select an InfoSet by Leader ID"
      default_option["selected"] = ""
      path_select.append(default_option)
      for iss_leader in self.iss_groups_to_ids:
        if len(self.iss_groups_to_ids[iss_leader]) > 1:
          option = soup.new_tag("option", value=iss_leader)
          option.string = f"InfoSet Lead Node: {iss_leader}"
          path_select.append(option)
      node_selector_row = select_menu_div.find("div", class_="row no-gutters")  # pylint: disable=attribute-error
      if node_selector_row:
        node_selector_row.insert_after(path_row)
    # %% Overwrite Dropdown Limit

    # 4.1. Inject `maxOptions: Infinity` for ALL TomSelect instances
    script_tags = soup.find_all("script")
    for script_tag in script_tags:
      # pylint: disable=attribute-error
      if script_tag.string and "new TomSelect" in script_tag.string:
        tomselect_replace_pattern = re.compile(
            r"(new TomSelect\(.+?,\s*\{.*?create:\s*false,)", re.DOTALL
        )
        if (
            tomselect_replace_pattern.search(script_tag.string)
            and "maxOptions: Infinity," not in script_tag.string
        ):
          script_tag.string = tomselect_replace_pattern.sub(
              r"\1\n                      maxOptions: Infinity,",
              script_tag.string,
          )
      # pylint: enable=attribute-error

    modified_html_content = str(soup)

    return modified_html_content

  def display_html(self, html_content: str):
    html_widget = widgets.HTML(value=html_content)
    display_html.display(html_widget)

    if output:
      iframe_height = 2000
      iframe_height_js = (
          "google.colab.output.setIframeHeight(0, true, "
          + f"{{maxHeight: {iframe_height}}})"
      )
      display_html.display(display_html.Javascript(iframe_height_js))

  def show_tree(self):
    html_content = self.html_from_nx()
    self._html_content = html_content
    self.display_html(html_content)

  def get_html_content(self) -> str:
    return self._html_content

  def set_html_content(self, html_content: str):
    self._html_content = html_content
