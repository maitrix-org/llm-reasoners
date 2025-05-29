#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <core/array.h>
#include <core/random.h>

using namespace core;

namespace py = pybind11;
using namespace py::literals;

static constexpr unsigned int RESERVED_INDICES[] = { 0 };
static const py::bool_ py_true(true);

template<typename T>
inline const T& choice(const T* items, unsigned int length) {
	return sample_uniform(items, length);
}

inline unsigned int randrange(unsigned int start, unsigned int end) {
	return start + sample_uniform(end - start);
}

inline unsigned int randrange(unsigned int end) {
	return sample_uniform(end);
}

struct node {
	unsigned int id;
	core::array<node*> children;
	core::array<node*> parents;

	node(unsigned int id) : id(id), children(8), parents(8) { }

	static inline void free(node& n) {
		core::free(n.children);
		core::free(n.parents);
	}
};

inline bool init(node& n, unsigned int id) {
	if (!array_init(n.children, 8)) {
		return false;
	} else if (!array_init(n.parents, 8)) {
		core::free(n.children);
		return false;
	}
	n.id = id;
	return true;
}

inline bool operator == (const node& first, const node& second) {
	return first.id == second.id;
}

bool generate_graph(array<node>& vertices, node*& start, node*& end, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id)
{
	if (!vertices.ensure_capacity(num_vertices))
		return false;
	for (unsigned int i = 0; i < num_vertices; i++)
		if (!init(vertices[i], i)) return false;
	vertices.length = num_vertices;

	/* sample a random DAG */
	for (unsigned int i = 1; i < num_vertices; i++) {
		/* sample the number of parent vertices */
		unsigned int num_parents;
		if (randrange(2) == 0)
			num_parents = 1;
		else
			num_parents = randrange(1, max_num_parents);
		num_parents = min(num_parents, i);

		array<unsigned int> available_parent_ids(i);
		for (unsigned int j = 0; j < i; j++)
			available_parent_ids.add(j);
		for (unsigned int j = 0; j < num_parents; j++) {
			unsigned int u = randrange(available_parent_ids.length);
			unsigned int parent_id = available_parent_ids[u];
			vertices[parent_id].children.add(&vertices[i]);
			vertices[i].parents.add(&vertices[parent_id]);
			available_parent_ids.remove(u);
		}
	}

	/* remove any correlation between graph topology and vertex IDs by shuffling the vertices */
	unsigned int* new_indices = (unsigned int*) alloca(sizeof(unsigned int) * (max_vertex_id + 1));
	for (unsigned int i = 0; i < max_vertex_id + 1; i++) new_indices[i] = i;
	shuffle(new_indices, max_vertex_id + 1);
	unsigned int src_index = 0;
	for (unsigned int i = 0; i < vertices.length; i++) {
		bool is_reserved = false;
		for (unsigned int j = 0; j < array_length(RESERVED_INDICES); j++) {
			if (new_indices[src_index] == RESERVED_INDICES[j]) {
				is_reserved = true;
				break;
			}
		}
		if (is_reserved)
			src_index++;
		vertices[i].id = new_indices[src_index];
		src_index++;
	}

	/* randomly select two vertices */
	start = &vertices[randrange(vertices.length - 1)];
	do {
		end = &vertices[randrange(vertices.length - 1)];
	} while (end == start);
	return true;
}

/* computes the number of lookahead steps to find the answer */
unsigned int lookahead_depth(const node* vertex, const node* next_vertex, const node* goal)
{
	array<pair<const node*, const node*>> frontier(8);
	for (const node* v : vertex->children)
		frontier.add(make_pair(v, v));
	array<const node*> visited(16);
	visited.append(vertex->children.data, vertex->children.length);
	unsigned int lookahead = 0;
	while (frontier.length != 0) {
		bool frontier_is_next_vertex = true;
		for (pair<const node*, const node*>& entry : frontier) {
			if (entry.value != next_vertex) {
				frontier_is_next_vertex = false;
				break;
			}
		}
		if (frontier_is_next_vertex)
			return lookahead;

		lookahead++;
		array<pair<const node*, const node*>> new_frontier(8);
		for (const pair<const node*, const node*>& entry : frontier) {
			const node* v = entry.key;
			const node* branch = entry.value;

			if (v == goal)
				return lookahead;
			for (const node* child : v->children) {
				if (!visited.contains(child)) {
					new_frontier.add(make_pair(child, branch));
					visited.add(child);
				} else if (branch == next_vertex) {
					for (unsigned int i = 0; i < new_frontier.length; i++)
						if (new_frontier[i].key == child)
							new_frontier[i].value = branch;
				}
			}
		}
		core::swap(frontier, new_frontier);
	}
	return lookahead;
}

void get_descendants(const node& vertex, array<const node*>& descendants) {
	array<const node*> queue(8);
	array<const node*> visited(16);
	queue[0] = &vertex;
	queue.length = 1;
	while (queue.length != 0) {
		const node* current = queue.pop();
		visited.add(current);
		for (const node* child : current->children) {
			if (!descendants.contains(child))
				descendants.add(child);
			if (visited.contains(child))
				continue;
			queue.add(child);
		}
	}
}

bool has_cycles(array<node>& vertices) {
	for (const node& vertex : vertices) {
		array<const node*> descendants(8);
		get_descendants(vertex, descendants);
		if (descendants.contains(&vertex))
			return true;
	}
	return false;
}

bool generate_graph_with_lookahead(array<node>& vertices, node*& start, node*& end, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id, unsigned int lookahead, unsigned int num_paths, unsigned int max_prefix_vertices)
{
	num_vertices = std::max(std::max(2u, num_vertices), 1 + num_paths * lookahead);

	if (!vertices.ensure_capacity(num_vertices))
		return false;
	for (unsigned int i = 0; i < num_vertices; i++)
		if (!init(vertices[i], i)) return false;
	vertices.length = num_vertices;

	vertices[1].parents.add(&vertices[0]);
	vertices[0].children.add(&vertices[1]);
	for (unsigned int i = 1; i < lookahead; i++) {
		vertices[1 + i].parents.add(&vertices[i]);
		vertices[i].children.add(&vertices[1 + i]);
	}
	unsigned int index;
	if (lookahead == 0) {
		index = 2;
	} else {
		index = 1 + lookahead;
		for (unsigned int j = 0; j < num_paths - 1; j++) {
			vertices[index].parents.add(&vertices[0]);
			vertices[0].children.add(&vertices[index]);
			index++;
			unsigned int other_branch_length = lookahead + randrange(std::min(2u, num_vertices - index - (num_paths - j - 1) * lookahead + 2));
			for (unsigned int i = 1; i < other_branch_length; i++) {
				vertices[index].parents.add(&vertices[index - 1]);
				vertices[index - 1].children.add(&vertices[index]);
				index++;
			}
		}
	}

	unsigned int num_prefix_vertices = randrange(min(max_prefix_vertices + 1, num_vertices - index + 1));
	node* prev_vertex = &vertices[0];
	for (unsigned int i = 0; i < num_prefix_vertices; i++) {
		vertices[index].children.add(prev_vertex);
		prev_vertex->parents.add(&vertices[index]);
		prev_vertex = &vertices[index];
		index++;
	}

	start = &vertices[0];
	end = &vertices[std::max(1u, lookahead)];

	/* sample some parent/ancestor vertices */
	constexpr float ALPHA = 0.5f;
	unsigned int* in_degrees = (unsigned int*) calloc(num_vertices, sizeof(unsigned int));
	unsigned int* out_degrees = (unsigned int*) calloc(num_vertices, sizeof(unsigned int));
	if (in_degrees == nullptr || out_degrees == nullptr) {
		if (in_degrees != nullptr) free(in_degrees);
		return false;
	}
	for (unsigned int i = 0; i < num_vertices; i++) {
		in_degrees[i] = vertices[i].parents.length;
		out_degrees[i] = vertices[i].children.length;
	}
	for (unsigned int i = index; i < num_vertices; i++) {
		/* sample the number of child and parent vertices */
		unsigned int num_children = randrange(0, max_num_parents);
		unsigned int num_parents = randrange(num_children == 0 ? 1 : 0, max_num_parents);
		num_children = std::min(num_children, i);
		num_parents = std::min(num_parents, i);

		/* sample the children of this new node */
		array<float> probabilities(index);
		float total_probability = 0.0f;
		for (unsigned int j = 0; j < index; j++) {
			probabilities[j] = ALPHA + in_degrees[j];
			total_probability += probabilities[j];
		}
		probabilities.length = index;

		array<unsigned int> sampled_children(std::max(1u, num_children));
		for (unsigned int j = 0; j < num_children; j++) {
			unsigned int u = sample_categorical(probabilities.data, total_probability, probabilities.length);
			sampled_children.add(u);
			total_probability -= probabilities[u];
			probabilities[u] = 0.0f;
		}

		for (unsigned int child_id : sampled_children) {
			vertices[index].children.add(&vertices[child_id]);
			vertices[child_id].parents.add(&vertices[index]);
			in_degrees[child_id] += 1;
		}

		/* sample the parents of this new node */
		total_probability = 0.0f;
		for (unsigned int j = 0; j < index; j++) {
			probabilities[j] = ALPHA + out_degrees[j];
			total_probability += probabilities[j];
		}

		/* to avoid creating a cycle, we have to remove any descendants from the possible parents */
		array<const node*> descendants(8);
		get_descendants(vertices[index], descendants);
		for (const node* descendant : descendants) {
			total_probability -= probabilities[descendant->id];
			probabilities[descendant->id] = 0.0f;
		}
		if (total_probability != 0.0f) {
			num_parents = std::min(num_parents,  index - (unsigned int) descendants.length);

			array<unsigned int> sampled_parents(std::max(1u, num_parents));
			for (unsigned int j = 0; j < num_parents; j++) {
				unsigned int u = sample_categorical(probabilities.data, total_probability, probabilities.length);
				sampled_parents.add(u);
				total_probability -= probabilities[u];
				probabilities[u] = 0.0f;
			}

			for (unsigned int parent_id : sampled_parents) {
				vertices[parent_id].children.add(&vertices[index]);
				vertices[index].parents.add(&vertices[parent_id]);
				out_degrees[parent_id] += 1;
			}
		}
		index += 1;
	}
	free(in_degrees);
	free(out_degrees);

	/* remove any correlation between graph topology and vertex IDs by shuffling the vertices */
	unsigned int* new_indices = (unsigned int*) alloca(sizeof(unsigned int) * (max_vertex_id + 1));
	for (unsigned int i = 0; i < max_vertex_id + 1; i++) new_indices[i] = i;
	shuffle(new_indices, max_vertex_id + 1);
	unsigned int src_index = 0;
	for (unsigned int i = 0; i < vertices.length; i++) {
		bool is_reserved = false;
		for (unsigned int j = 0; j < array_length(RESERVED_INDICES); j++) {
			if (new_indices[src_index] == RESERVED_INDICES[j]) {
				is_reserved = true;
				break;
			}
		}
		if (is_reserved)
			src_index++;
		vertices[i].id = new_indices[src_index];
		src_index++;
	}
	return true;
}

bool generate_example(array<node>& vertices, node*& start, node*& end, array<array<node*>>& paths, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id, bool get_shortest_paths, int lookahead, unsigned int num_paths, unsigned int max_prefix_vertices)
{
	if (lookahead == -1) {
		if (!generate_graph(vertices, start, end, num_vertices, max_num_parents, max_vertex_id))
			return false;
	} else {
		if (!generate_graph_with_lookahead(vertices, start, end, num_vertices, max_num_parents, max_vertex_id, lookahead, num_paths, max_prefix_vertices))
			return false;
	}

	/* find the shortest paths from `start` to `end` */
	array<pair<node*, unsigned int>> queue(16);
	queue[0].key = start;
	queue[0].value = 0;
	queue.length = 1;
	array_map<node*, array_map<node*, unsigned int>> reverse_pointers(16);
	while (queue.length != 0) {
		pair<node*, unsigned int> item = queue.pop();
		node* current = item.key;
		unsigned int distance = item.value;

		for (node* child : current->children) {
			if (!reverse_pointers.ensure_capacity(reverse_pointers.size + 1)) {
				for (const auto& entry : reverse_pointers) core::free(entry.value);
				return false;
			}
			bool contains;
			array_map<node*, unsigned int>& value = reverse_pointers.get(child, contains);
			if (!contains) {
				if (!array_map_init(value, 4)) {
					for (const auto& entry : reverse_pointers) core::free(entry.value);
					return false;
				}
				reverse_pointers.keys[reverse_pointers.size++] = child;
			}

			if (!value.ensure_capacity(value.size + 1)) {
				for (const auto& entry : reverse_pointers) core::free(entry.value);
				return false;
			}
			unsigned int& distance_value = value.get(current, contains);
			if (!contains) {
				distance_value = distance + 1;
				value.keys[value.size++] = current;
			} else if (distance_value > distance + 1) {
				distance_value = distance + 1;
			} else {
				continue;
			}

			bool found_child = false;
			for (unsigned int j = 0; j < queue.length; j++) {
				if (queue[j].key == child) {
					queue[j].value = std::min(queue[j].value, distance + 1);
					found_child = true;
					break;
				}
			}
			if (!found_child)
				queue.add(make_pair(child, distance + 1));
		}
	}

	if (!reverse_pointers.contains(end)) {
		for (const auto& entry : reverse_pointers) core::free(entry.value);
		return false;
	}

	array_map<const node*, array<node*>> forward_pointers(16);
	array<node*> dist_queue(16);
	dist_queue[0] = end;
	dist_queue.length = 1;
	while (dist_queue.length != 0) {
		node* current = dist_queue.pop();
		if (current == start)
			continue;

		array<node*> prev_nodes(8);
		const array_map<node*, unsigned int>& value = reverse_pointers.get(current);
		if (get_shortest_paths) {
			unsigned int min_distance = value.values[0];
			for (unsigned int i = 1; i < value.size; i++)
				if (value.values[i] < min_distance) min_distance = value.values[i];

			for (unsigned int i = 0; i < value.size; i++)
				if (value.values[i] == min_distance) prev_nodes.add(value.keys[i]);
		} else {
			prev_nodes.append(value.keys, value.size);
		}
		for (const node* prev : prev_nodes) {
			if (!forward_pointers.ensure_capacity(forward_pointers.size + 1)) {
				for (const auto& entry : reverse_pointers) core::free(entry.value);
				for (const auto& entry : forward_pointers) core::free(entry.value);
				return false;
			}
			bool contains;
			array<node*>& fptrs = forward_pointers.get(prev, contains);
			if (!contains) {
				if (!array_init(fptrs, 4)) {
					for (const auto& entry : reverse_pointers) core::free(entry.value);
					for (const auto& entry : forward_pointers) core::free(entry.value);
					return false;
				}
				forward_pointers.keys[forward_pointers.size++] = prev;
			}
			fptrs.add(current);
		}
		dist_queue.append(prev_nodes.data, prev_nodes.length);
	}

	/* free `reverse_pointers` */
	for (const auto& entry : reverse_pointers) core::free(entry.value);

	/* construct the shortest paths from the forward pointers */
	array<array<node*>> path_queue(8);
	if (!array_init(path_queue[0], 1)) {
		for (const auto& entry : forward_pointers) core::free(entry.value);
		return false;
	}
	path_queue[0].add(start);
	path_queue.length = 1;
	while (path_queue.length != 0) {
		array<node*>& partial_path = *((array<node*>*) alloca(sizeof(array<node*>)));
		core::move(path_queue.last(), partial_path);
		path_queue.length--;

		if (partial_path.last() == end) {
			if (!paths.ensure_capacity(paths.length + 1)) {
				for (const auto& entry : forward_pointers) core::free(entry.value);
				for (auto& p : path_queue) core::free(p);
				core::free(partial_path);
				return false;
			}
			core::move(partial_path, paths[paths.length++]);
			if (paths.length > 64) {
				for (const auto& entry : forward_pointers) core::free(entry.value);
				for (auto& p : path_queue) core::free(p);
				return false;
			}
			continue;
		}
		for (node* next : forward_pointers.get(partial_path.last())) {
			if (!path_queue.ensure_capacity(path_queue.length + 1)
			 || !array_init(path_queue[path_queue.length], partial_path.length + 1))
			{
				for (const auto& entry : forward_pointers) core::free(entry.value);
				for (auto& p : path_queue) core::free(p);
				core::free(partial_path);
				return false;
			}
			path_queue.length++;
			path_queue.last().append(partial_path.data, partial_path.length);
			path_queue.last().add(next);
		}
		core::free(partial_path);
	}

	for (const auto& entry : forward_pointers) core::free(entry.value);
	return true;
}

bool has_path(const node* start, const node* end)
{
	array<const node*> stack(8);
	stack[0] = start;
	stack.length = 1;
	array<const node*> visited(16);
	while (stack.length != 0) {
		const node* v = stack.pop();
		if (v == end)
			return true;
		for (node* child : v->children) {
			if (!visited.contains(child)) {
				visited.add(child);
				stack.add(child);
			}
		}
	}
	return false;
}

py::tuple generate_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const int max_lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const int max_prefix_vertices, const bool quiet=false, const int num_paths=2)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	unsigned int ntokens = (max_input_size - 5) / 3 + 5;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, ntokens};
	size_t label_shape[1]{dataset_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	py::array_t<int64_t, py::array::c_style> labels(label_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	auto labels_mem = labels.mutable_unchecked<1>();
	unsigned int* lookahead_step_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	unsigned int* path_length_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	for (unsigned int i = 0; i < max_input_size; i++) {
		lookahead_step_histogram[i] = 0;
		path_length_histogram[i] = 0;
	}
	float* MAX_FREQS_PER_BUCKET = (float*) alloca(sizeof(float) * max_input_size);
	if (max_lookahead == -1) {
		for (unsigned int i = 0; i < max_input_size; i++)
			MAX_FREQS_PER_BUCKET[i] = 1.0;
	} else {
		for (unsigned int i = 0; i < (unsigned) max_lookahead + 1; i++)
			MAX_FREQS_PER_BUCKET[i] = 1.0 / (max_lookahead+1);
		for (unsigned int i = max_lookahead + 1; i < max_input_size; i++)
			MAX_FREQS_PER_BUCKET[i] = 0.0;
		MAX_FREQS_PER_BUCKET[max_lookahead] += 0.05;
	}

	unsigned int* potential_lookaheads = (unsigned int*) alloca(max((size_t) 1, sizeof(unsigned int) * (max_lookahead + 1)));
	unsigned int potential_lookahead_count = 0;
	while (num_generated < dataset_size) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			if (max_lookahead == -1) {
				unsigned int num_vertices = randrange(3, (max_input_size - 5) / 3);
				if (!generate_example(g, start, end, paths, num_vertices, 4, (max_input_size - 5) / 3, true, -1, 0, max_prefix_vertices == -1 ? max_input_size : max_prefix_vertices)) {
					for (node& n : g) core::free(n);
					for (array<node*>& a : paths) core::free(a);
					g.length = 0; paths.length = 0;
					continue;
				}
			} else {
				potential_lookahead_count = 0;
				for (unsigned int i = 0; i < (unsigned) max_lookahead + 1; i++)
					if (num_generated == 0 || lookahead_step_histogram[i] / num_generated < MAX_FREQS_PER_BUCKET[i])
						potential_lookaheads[potential_lookahead_count++] = i;
//				unsigned int lookahead = choice(potential_lookaheads, potential_lookahead_count);
                unsigned int lookahead = max_lookahead;

//				unsigned int num_paths;
//				if (lookahead == 0) {
//					num_paths = randrange(1, 3);
//				} else {
//					unsigned int max_num_paths = (max_edges - 1) / lookahead;
//					num_paths = randrange(2, max_num_paths + 1);
//				}

				unsigned int num_vertices = std::min(std::min(lookahead * num_paths + 1 + randrange(0, 6), (max_input_size-5) / 3), max_edges + 1);
				if (!generate_example(g, start, end, paths, num_vertices, 4, (max_input_size - 5) / 3, true, lookahead, num_paths, max_prefix_vertices == -1 ? max_input_size : max_prefix_vertices)) {
					for (node& n : g) core::free(n);
					for (array<node*>& a : paths) core::free(a);
					g.length = 0; paths.length = 0;
					continue;
				}

				unsigned int shortest_path_length = paths[0].length;
                for (unsigned int i = 1; i < paths.length; i++)
                    if (paths[i].length < shortest_path_length)
                        shortest_path_length = paths[i].length;

                // Count how many paths have the shortest length.
                unsigned int shortest_path_count = 0;
                for (unsigned int i = 0; i < paths.length; i++) {
                    if (paths[i].length == shortest_path_length)
                        shortest_path_count++;
                }

                // Only accept graphs with a unique (one) shortest path.
                if (shortest_path_length <= 1 || shortest_path_count != 1) {
                    for (node& n : g) core::free(n);
                    for (array<node*>& a : paths) core::free(a);
                    g.length = 0; paths.length = 0;
                    continue;
                }
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				unsigned int lookahead_steps = lookahead_depth(path[j-1], path[j], end);
				array<node*> useful_steps(8);
				for (node* v : path[j-1]->children)
					if (has_path(v, end)) useful_steps.add(v);

				/* check if this input is reserved */
				py::object contains = reserved_inputs.attr("__contains__");
				py::tuple example_tuple(example.length);
				for (unsigned int i = 0; i < example.length; i++)
					example_tuple[i] = example[i];
				if (contains(example_tuple).is(py_true)) {
					num_collisions += 1;
					continue;
				}

				if (num_generated != 0 && lookahead_step_histogram[lookahead_steps] / num_generated >= MAX_FREQS_PER_BUCKET[lookahead_steps])
					continue;
				lookahead_step_histogram[lookahead_steps] += 1;
				path_length_histogram[j] += 1;

				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					inputs_mem(num_generated, i) = PADDING_TOKEN;
				for (unsigned int i = 0; i < example.length; i++)
					inputs_mem(num_generated, max_input_size - example.length + i) = example[i];
				for (unsigned int i = 0; i < ntokens; i++)
					outputs_mem(num_generated, i) = 0.0f;
				for (unsigned int i = 0; i < useful_steps.length; i++)
					outputs_mem(num_generated, useful_steps[i]->id) = 1.0f;
				labels_mem(num_generated) = choice(useful_steps.data, useful_steps.length)->id;
				num_generated++;
				if (num_generated == dataset_size)
					break;
			}
			if (num_generated == dataset_size)
				break;
		}

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= dataset_size)) {
			printf("%d examples generated.\n", num_generated);

			printf("Lookahead steps histogram:\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (lookahead_step_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) lookahead_step_histogram[i] / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");

			printf("Path length histogram:\n");
			printf("[");
			first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (path_length_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) path_length_histogram[i] / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");
			fflush(stdout);
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, labels, num_collisions);
}

py::array_t<int64_t, py::array::c_style> lookahead_histogram(const unsigned int max_input_size, const uint64_t num_samples, const unsigned int max_edges, const int distance_from_start, const bool quiet=false)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int max_lookahead = ((max_input_size - 5) / 3 - 1) / 2;
	size_t histogram_shape[1]{max_lookahead};
	py::array_t<int64_t, py::array::c_style> histogram(histogram_shape);
	auto histogram_mem = histogram.mutable_unchecked<1>();
	for (unsigned int i = 0; i < max_lookahead; i++)
		histogram_mem(i) = 0;

	while (num_generated < num_samples) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			unsigned int num_vertices = randrange(3, (max_input_size - 5) / 3);
			if (!generate_example(g, start, end, paths, num_vertices, 4, (max_input_size - 5) / 3, true, -1, 0, -1)) {
				for (node& n : g) core::free(n);
				for (array<node*>& a : paths) core::free(a);
				g.length = 0; paths.length = 0;
				continue;
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				unsigned int lookahead_steps = lookahead_depth(path[j-1], path[j], end);
				histogram_mem(lookahead_steps) += 1;
				num_generated++;
				if (num_generated == num_samples)
					break;
			}
			if (num_generated == num_samples)
				break;
		}

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= num_samples)) {
			printf("%d examples generated.\n", num_generated);

			printf("Lookahead steps histogram:\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_lookahead; i++) {
				if (histogram_mem(i) == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) histogram_mem(i) / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

	return histogram;
}

py::tuple generate_reachable_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const unsigned int lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const int reachable_distance, const unsigned int start_vertex_index, const bool exclude_start_vertex)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, max_input_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	py::list valid_outputs;

	unsigned int max_vertex_id = (max_input_size - 5) / 3;
	while (num_generated < dataset_size) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			unsigned int num_paths;
			if (lookahead == 0) {
				num_paths = randrange(1, 3);
			} else {
				unsigned int max_num_paths = (max_edges - 1) / lookahead;
				num_paths = randrange(2, max_num_paths + 1);
			}

			unsigned int num_vertices = std::min(std::min(lookahead * num_paths + 1 + randrange(0, 6), (max_input_size-5) / 3), max_edges + 1);
			if (!generate_example(g, start, end, paths, num_vertices, 4, max_vertex_id, true, lookahead, num_paths, -1)) {
				for (node& n : g) core::free(n);
				for (array<node*>& a : paths) core::free(a);
				g.length = 0; paths.length = 0;
				continue;
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				/* compute the set of reachable vertices */
				node** vertex_id_map = (node**) calloc(max_vertex_id + 1, sizeof(node*));
				for (unsigned int i = 0; i < g.length; i++)
					vertex_id_map[g[i].id] = &g[i];
				array<unsigned int> reachable(16);
				array<pair<unsigned int, unsigned int>> stack(16);
				unsigned int start_vertex;
				if (example.length < start_vertex_index)
					start_vertex = start->id;
				else start_vertex = example[example.length - start_vertex_index];
				stack.add(make_pair(start_vertex, 0u));
				while (stack.length != 0) {
					pair<unsigned int, unsigned int> entry = stack.pop();
					unsigned int current_vertex = entry.key;
					unsigned int current_distance = entry.value;
					if (!reachable.contains(current_vertex))
						reachable.add(current_vertex);
					if (reachable_distance > 0 && current_distance + 1 <= (unsigned int) reachable_distance) {
						for (node* child : vertex_id_map[current_vertex]->children)
							stack.add(make_pair(child->id, current_distance + 1));
					} else if (reachable_distance < 0 && current_distance + 1 <= (unsigned int) -reachable_distance) {
						for (node* parent : vertex_id_map[current_vertex]->parents)
							stack.add(make_pair(parent->id, current_distance + 1));
					}
				}
				if (exclude_start_vertex)
					reachable.remove(reachable.index_of(start_vertex));

				array<node*> useful_steps(8);
				for (node* v : path[j-1]->children)
					if (has_path(v, end)) useful_steps.add(v);

				/* check if this input is reserved */
				py::object contains = reserved_inputs.attr("__contains__");
				py::tuple example_tuple(example.length);
				for (unsigned int i = 0; i < example.length; i++)
					example_tuple[i] = example[i];
				if (contains(example_tuple).is(py_true)) {
					num_collisions += 1;
					continue;
				}

				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					inputs_mem(num_generated, i) = PADDING_TOKEN;
				for (unsigned int i = 0; i < example.length; i++)
					inputs_mem(num_generated, max_input_size - example.length + i) = example[i];
				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					outputs_mem(num_generated, i) = 0;
				for (unsigned int i = 0; i < example.length; i++)
					outputs_mem(num_generated, max_input_size - example.length + i) = reachable.contains(example[i]) ? 1 : 0;
				num_generated++;
				if (num_generated == dataset_size)
					break;
			}
			if (num_generated == dataset_size)
				break;
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, num_collisions);
}

bool generate_dfs_example(array<node>& vertices, const node*& start, const node*& end, unsigned int& current_node_index, array<const node*>& path, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id, const int requested_backtrack, unsigned int max_edges)
{
	if (!vertices.ensure_capacity(num_vertices))
		return false;
	for (unsigned int i = 0; i < num_vertices; i++)
		if (!init(vertices[i], i)) return false;
	vertices.length = num_vertices;

	/* sample some parent/ancestor vertices */
	constexpr float ALPHA = 1.0f;
	unsigned int* out_degrees = (unsigned int*) calloc(num_vertices, sizeof(unsigned int));
	if (out_degrees == nullptr)
		return false;
	for (unsigned int i = 1; i < num_vertices; i++) {
		/* sample the number of child and parent vertices */
		unsigned int num_parents = randrange(1, max_num_parents);
		num_parents = std::min(num_parents, i);

		/* sample the parents of this new node */
		float total_probability = 0.0f;
		array<float> probabilities(i);
		for (unsigned int j = 0; j < i; j++) {
			probabilities[j] = ALPHA + out_degrees[j];
			total_probability += probabilities[j];
		}
		probabilities.length = i;

		array<unsigned int> sampled_parents(std::max(1u, num_parents));
		for (unsigned int j = 0; j < num_parents; j++) {
			unsigned int u = sample_categorical(probabilities.data, total_probability, probabilities.length);
			sampled_parents.add(u);
			total_probability -= probabilities[u];
			probabilities[u] = 0.0f;
		}

		for (unsigned int parent_id : sampled_parents) {
			vertices[parent_id].children.add(&vertices[i]);
			vertices[i].parents.add(&vertices[parent_id]);
			out_degrees[parent_id] += 1;
		}
	}
	free(out_degrees);

	/* create a list of edges that we can remove */
	unsigned int new_edge_count = 0;
	array<pair<unsigned int, unsigned int>> all_edges(16);
	for (const node& vertex : vertices)
		for (const node* child : vertex.children)
			all_edges.add(make_pair(vertex.id, child->id));
	unsigned int total_edge_count = all_edges.length;

	array<pair<unsigned int, unsigned int>> removable_edges(16);
	removable_edges.append(all_edges.data, all_edges.length);
	while (true) {
		/* select a start and goal vertex uniformly at random */
		unsigned int start_index;
		unsigned int end_index;
		if (requested_backtrack == 0 || requested_backtrack == -1) {
			start_index = randrange(vertices.length - 1);
			end_index = start_index + 1 + randrange(vertices.length - start_index - 1);
		} else {
			/* if `requested_backtrack` is specified, divide the vertices after the
			   start vertex into two subgraphs: the first subgraph will contain the goal,
			   and the second subgraph must have size at least `requested_backtrack` */
			const node* second_subgraph_root = &vertices[vertices.length - requested_backtrack];
			if (second_subgraph_root->parents.length == 1 && second_subgraph_root->parents[0]->id == second_subgraph_root->id - 1) {
				start_index = randrange(vertices.length - requested_backtrack - 1);
			} else {
				do {
					start_index = choice(second_subgraph_root->parents.data, second_subgraph_root->parents.length)->id;
				} while (start_index == second_subgraph_root->id - 1);
			}
			end_index = start_index + 1 + randrange(vertices.length - start_index - requested_backtrack - 1);
		}
		start = &vertices[start_index];
		end = &vertices[end_index];

		if (requested_backtrack != 0 && requested_backtrack != -1) {
			/* if `requested_backtrack` is specified, make sure there is a path from the
			   start vertex to all vertices in the first and second subgraphs */
			for (unsigned int i = start_index + 1; i < vertices.length; i++) {
				bool has_path_to_start = false;
				for (const node* parent : vertices[i].parents) {
					if (i >= vertices.length - requested_backtrack && (parent->id == start_index || parent->id >= vertices.length - requested_backtrack)) {
						has_path_to_start = true;
						break;
					} else if (i < vertices.length - requested_backtrack && parent->id >= start_index) {
						has_path_to_start = true;
						break;
					}
				}
				if (!has_path_to_start) {
					vertices[start_index].children.add(&vertices[i]);
					vertices[i].parents.add(&vertices[start_index]);
					new_edge_count++;
					total_edge_count++;
				}
			}
		}

		/* perform DFS from the start vertex */
		array<pair<const node*, const node*>> queue(8);
		if (requested_backtrack == 0 || requested_backtrack == -1) {
			queue[0] = make_pair((const node*) nullptr, start);
			queue.length = 1;
		} else {
			/* if `requested_backtrack` is specified, start the DFS in the second subgraph */
			path.add(start);
			const node** first_subgraph_children = (const node**) malloc(sizeof(const node*) * start->children.length);
			const node** second_subgraph_children = (const node**) malloc(sizeof(const node*) * start->children.length);
			unsigned int num_first_subgraph_children = 0;
			unsigned int num_second_subgraph_children = 0;
			for (const node* child : start->children) {
				if (child->id >= vertices.length - requested_backtrack) {
					second_subgraph_children[num_second_subgraph_children++] = child;
				} else {
					first_subgraph_children[num_first_subgraph_children++] = child;
				}
			}
			shuffle(first_subgraph_children, num_first_subgraph_children);
			shuffle(second_subgraph_children, num_second_subgraph_children);
			for (unsigned int i = 0; i < num_first_subgraph_children; i++)
				queue.add(make_pair(start, first_subgraph_children[i]));
			for (unsigned int i = 0; i < num_second_subgraph_children; i++)
				queue.add(make_pair(start, second_subgraph_children[i]));
			free(first_subgraph_children);
			free(second_subgraph_children);
		}
		bool found_goal = false;
		while (queue.length != 0) {
			pair<const node*, const node*> state = queue.pop();
			const node* parent = state.key;
			const node* current = state.value;
			if (path.contains(current))
				continue;
			path.add(current);
			if (parent != nullptr) {
				unsigned int index = removable_edges.index_of(make_pair(parent->id, current->id));
				if (index < removable_edges.length)
					removable_edges.remove(index);
			}

			if (current == end) {
				found_goal = true;
				break;
			} else if (current->children.contains(end)) {
				found_goal = true;
				path.add(end);
				unsigned int index = removable_edges.index_of(make_pair(current->id, end->id));
				if (index < removable_edges.length)
					removable_edges.remove(index);
				break;
			}

			if (current->children.length == 0)
				continue;
			const node** children = (const node**) malloc(sizeof(const node*) * current->children.length);
			for (unsigned int i = 0; i < current->children.length; i++)
				children[i] = current->children[i];
			shuffle(children, current->children.length);
			for (unsigned int i = 0; i < current->children.length; i++) {
				if (path.contains(children[i])) continue;
				queue.add(make_pair(current, children[i]));
			}
			free(children);
		}

		/* check if the goal vertex is reachable from the start vertex */
		if (found_goal)
			break;
		removable_edges.clear();
		removable_edges.append(all_edges.data, all_edges.length);
		path.clear();
	}

	if (requested_backtrack == -1)
		current_node_index = randrange(path.length - 1);
	else
		current_node_index = requested_backtrack;

	/* remove edges to avoid generating a graph with too many edges */
	unsigned int edges_to_remove;
	if (total_edge_count > max_edges)
		edges_to_remove = total_edge_count - max_edges;
	else
		edges_to_remove = new_edge_count;
	edges_to_remove = min((unsigned int) removable_edges.length, edges_to_remove);
	for (unsigned int i = 0; i < edges_to_remove; i++) {
		unsigned int u = randrange(removable_edges.length);
		unsigned int parent_id = removable_edges[u].key;
		unsigned int child_id = removable_edges[u].value;
		vertices[parent_id].children.remove(vertices[parent_id].children.index_of(&vertices[child_id]));
		vertices[child_id].parents.remove(vertices[child_id].parents.index_of(&vertices[parent_id]));
		removable_edges.remove(u);
	}

	/* remove any correlation between graph topology and vertex IDs by shuffling the vertices */
	unsigned int* new_indices = (unsigned int*) alloca(sizeof(unsigned int) * (max_vertex_id + 1));
	for (unsigned int i = 0; i < max_vertex_id + 1; i++) new_indices[i] = i;
	shuffle(new_indices, max_vertex_id + 1);
	unsigned int src_index = 0;
	for (unsigned int i = 0; i < vertices.length; i++) {
		bool is_reserved = false;
		for (unsigned int j = 0; j < array_length(RESERVED_INDICES); j++) {
			if (new_indices[src_index] == RESERVED_INDICES[j]) {
				is_reserved = true;
				break;
			}
		}
		if (is_reserved)
			src_index++;
		vertices[i].id = new_indices[src_index];
		src_index++;
	}

	return true;
}

py::tuple generate_dfs_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const py::object& reserved_inputs, const int requested_backtrack, const bool random_padding, const bool uniform, const bool quiet=false)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int longest_path_length = (max_input_size - 4) / 4;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	unsigned int ntokens = (max_input_size - 5) / 3 + 5;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, ntokens};
	size_t label_shape[1]{dataset_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	py::array_t<int64_t, py::array::c_style> labels(label_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	auto labels_mem = labels.mutable_unchecked<1>();

	unsigned int* backtrack_distance_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	for (unsigned int i = 0; i < max_input_size; i++)
		backtrack_distance_histogram[i] = 0;

	float* MAX_FREQS_PER_BUCKET = (float*) alloca(sizeof(float) * max_input_size);
	if (requested_backtrack == -1) {
		for (unsigned int i = 0; i < max_input_size; i++)
			MAX_FREQS_PER_BUCKET[i] = 1.0;
	} else {
		for (unsigned int i = 0; i < (unsigned) requested_backtrack + 1; i++)
			MAX_FREQS_PER_BUCKET[i] = 1.0 / (requested_backtrack+1);
		for (unsigned int i = requested_backtrack + 1; i < max_input_size; i++)
			MAX_FREQS_PER_BUCKET[i] = 0.0;
		MAX_FREQS_PER_BUCKET[requested_backtrack] += 0.05;
	}

	array<const node*> path(32);
	unsigned int* potential_backtracks = (unsigned int*) alloca(max((size_t) 1, sizeof(unsigned int) * (requested_backtrack + 1)));
	unsigned int potential_backtrack_count = 0;
	unsigned int num_attempts = 0;
	while (num_generated < dataset_size) {
		if (num_attempts >= 10000000)
			break;
		num_attempts++;
		array<node> g(32);
		const node* start; const node* end;
		unsigned int current_node_index;
		while (true) {
			unsigned int num_vertices = std::max(2u, randrange(longest_path_length + 1));
			int backtrack = requested_backtrack;
			if (requested_backtrack != -1) {
				if (uniform) {
					potential_backtrack_count = 0;
					for (unsigned int i = 0; i < (unsigned) requested_backtrack + 1; i++)
						if (num_generated == 0 || backtrack_distance_histogram[i] / num_generated < MAX_FREQS_PER_BUCKET[i])
							potential_backtracks[potential_backtrack_count++] = i;
					backtrack = choice(potential_backtracks, potential_backtrack_count);
				}
				num_vertices = std::max((unsigned int) backtrack + 2, num_vertices);
			}
			if (!generate_dfs_example(g, start, end, current_node_index, path, num_vertices, max_input_size / 24 + 1, (max_input_size - 5) / 3, backtrack, longest_path_length)) {
				for (node& n : g) core::free(n);
				g.length = 0; path.length = 0;
				continue;
			}
			break;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > longest_path_length) {
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		/* randomly select a vertex in the DFS trace */
		path.length = current_node_index + 1;

		array<const node*> unvisited(4);
		unsigned int backtrack_distance = max_input_size - 1;
		for (unsigned int j = current_node_index + 1; j > 0; j--) {
			for (const node* child : path[j-1]->children) {
				if (!path.contains(child))
					unvisited.add(child);
			}
			if (unvisited.length != 0) {
				backtrack_distance = current_node_index + 1 - j;
				break;
			}
		}

		if (random_padding) {
			/* sample padding randomly until the path is the correct length */
			unsigned int* padding_lengths = (unsigned int*) calloc(path.length + 1, sizeof(unsigned int));
			unsigned int available_padding = longest_path_length - path.length;
			for (unsigned int j = path.length; j > 0; j--) {
				padding_lengths[j] = randrange(available_padding + 1);
				available_padding -= padding_lengths[j];
			}
			padding_lengths[0] = available_padding;

			for (unsigned int j = 0; j < path.length; j++) {
				for (unsigned int k = 0; k < padding_lengths[j]; k++)
					prefix[prefix.length++] = PATH_PREFIX_TOKEN;
				prefix[prefix.length++] = path[j]->id;
			}
			for (unsigned int k = 0; k < padding_lengths[path.length]; k++)
				prefix[prefix.length++] = PATH_PREFIX_TOKEN;
			unsigned int effective_backtrack = backtrack_distance;
			for (unsigned int j = path.length; j > 0; j--) {
				effective_backtrack += padding_lengths[j];
				if (path.length - j == backtrack_distance)
					break;
			}
			backtrack_distance = effective_backtrack;
			free(padding_lengths);
		} else {
			for (unsigned int j = path.length; j < longest_path_length; j++)
				prefix[prefix.length++] = PATH_PREFIX_TOKEN;
			for (unsigned int j = 0; j < path.length; j++)
				prefix[prefix.length++] = path[j]->id;
		}

		if ((requested_backtrack != -1 && !uniform && (unsigned int) requested_backtrack != backtrack_distance)
		 || (uniform && num_generated != 0 && backtrack_distance_histogram[backtrack_distance] / num_generated >= MAX_FREQS_PER_BUCKET[backtrack_distance]))
		{
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			continue;
		}

		/* check if this input is reserved */
		py::object contains = reserved_inputs.attr("__contains__");
		py::tuple example_tuple(prefix.length);
		for (unsigned int i = 0; i < prefix.length; i++)
			example_tuple[i] = prefix[i];
		if (contains(example_tuple).is(py_true)) {
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			num_collisions += 1;
			continue;
		}

		backtrack_distance_histogram[backtrack_distance]++;

		for (unsigned int i = 0; i < max_input_size - prefix.length; i++)
			inputs_mem(num_generated, i) = PADDING_TOKEN;
		for (unsigned int i = 0; i < prefix.length; i++)
			inputs_mem(num_generated, max_input_size - prefix.length + i) = prefix[i];
		for (unsigned int i = 0; i < ntokens; i++)
			outputs_mem(num_generated, i) = 0.0f;
		for (unsigned int i = 0; i < unvisited.length; i++)
			outputs_mem(num_generated, unvisited[i]->id) = 1.0f;
		labels_mem(num_generated) = path[current_node_index+1]->id;
		num_generated++;

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= dataset_size)) {
			printf("%d examples generated.\n", num_generated);
			fflush(stdout);

			printf("Backtrack distance histogram: (log frequencies)\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (backtrack_distance_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, log(backtrack_distance_histogram[i]) - log(num_generated));
				first = false;
			}
			printf("]\n");
		}

		for (node& n : g) core::free(n);
		g.length = 0; path.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, labels, num_collisions);
}

bool generate_si_example(array<node>& vertices, const node*& start, const node*& end, unsigned int& current_node_index, array<const node*>& path, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id, unsigned int max_edges, unsigned int frontier_size, unsigned int branch_size)
{
	if (!vertices.ensure_capacity(num_vertices))
		return false;
	for (unsigned int i = 0; i < num_vertices; i++)
		if (!init(vertices[i], i)) return false;
	vertices.length = num_vertices;

	/* the current vertex must have at least `branch_size` vertices following it */
	current_node_index = randrange(num_vertices - branch_size);
	node* current_node = &vertices[current_node_index];

	/* randomly sample the start and goal vertices */
	unsigned int start_index = randrange(min(current_node_index + 1, num_vertices - frontier_size));
	unsigned int end_index = randrange(max(current_node_index + 1, start_index + frontier_size), num_vertices);
	start = &vertices[start_index]; end = &vertices[end_index];

	/* sample some parent/ancestor vertices */
	constexpr float ALPHA = 1.0f;
	unsigned int* out_degrees = (unsigned int*) calloc(num_vertices, sizeof(unsigned int));
	if (out_degrees == nullptr)
		return false;
	for (unsigned int i = 1; i < num_vertices; i++) {
		/* sample the number of child and parent vertices */
		unsigned int num_parents = randrange(1, max_num_parents);
		num_parents = std::min(num_parents, i);

		/* sample the parents of this new node */
		float total_probability = 0.0f;
		array<float> probabilities(i);
		for (unsigned int j = 0; j < i; j++) {
			probabilities[j] = ALPHA + out_degrees[j];
			total_probability += probabilities[j];
		}
		probabilities.length = i;
		if (current_node->children.length == branch_size) {
			total_probability -= probabilities[current_node_index];
			probabilities[current_node_index] = 0.0f;
		}

		array<unsigned int> sampled_parents(std::max(1u, num_parents));
		for (unsigned int j = 0; j < num_parents; j++) {
			unsigned int u = sample_categorical(probabilities.data, total_probability, probabilities.length);
			sampled_parents.add(u);
			total_probability -= probabilities[u];
			probabilities[u] = 0.0f;
			if (total_probability < 1.0e-6f)
				break;
		}

		for (unsigned int parent_id : sampled_parents) {
			vertices[parent_id].children.add(&vertices[i]);
			vertices[i].parents.add(&vertices[parent_id]);
			out_degrees[parent_id] += 1;
		}
	}
	free(out_degrees);

	/* randomly sample the frontier (visited vertices with unvisited child nodes) */
	array<node*> frontier(8);
	array<node*> available_nodes(8);
	frontier.add(&vertices[start_index]);
	if (start_index != current_node_index)
		frontier.add(&vertices[current_node_index]);
	for (unsigned int i = start_index + 1; i < vertices.length - 1; i++)
		if (i != end_index && !frontier.contains(&vertices[i]))
			available_nodes.add(&vertices[i]);
	while (frontier.length < frontier_size) {
		unsigned int r = randrange(available_nodes.length);
		frontier.add(available_nodes[r]);
		available_nodes.remove(r);
	}

	unsigned int new_edge_count = 0;
	array<pair<unsigned int, unsigned int>> removable_edges(16);
	for (const node& vertex : vertices)
		for (const node* child : vertex.children)
			removable_edges.add(make_pair(vertex.id, child->id));
	unsigned int total_edge_count = removable_edges.length;

	/* construct a path such that all vertices in `frontier` are visited */
	array<pair<const node*, node*>> queue(8);
	for (node* child : start->children)
		if (frontier.contains(child)) queue.add(make_pair(start, child));
	while (queue.length != 0) {
		unsigned int r = randrange(queue.length);
		const node* prev = queue[r].key;
		const node* next = queue[r].value;
		queue.remove(r);
		if (path.contains(next))
			continue;
		path.add(prev);
		path.add(next);
		removable_edges.remove(removable_edges.index_of(make_pair(prev->id, next->id)));

		for (node* child : next->children)
			if (frontier.contains(child)) queue.add(make_pair(next, child));
	}

	/* get set of vertices reachable from `start` */
	array<node*> stack(8);
	array<node*> reachable(8);
	stack[stack.length++] = &vertices[start_index];
	while (stack.length != 0) {
		node* current = stack.pop();
		if (reachable.contains(current))
			continue;
		reachable.add(current);

		for (node* child : current->children) {
			if (!reachable.contains(child) && !stack.contains(child))
				stack.add(child);
		}
	}

	/* for the unvisited vertices in `frontier`, move edges until they have an incoming edge from a visited vertex */
	for (node* frontier_vertex : frontier) {
		if (frontier_vertex == start || path.index_of(frontier_vertex) % 2 == 1)
			continue;

		/* get the most recent visited ancestors */
		array<node*> stack(8);
		array<node*> visited_ancestors(8);
		stack.append(frontier_vertex->parents.data, frontier_vertex->parents.length);
		while (stack.length != 0) {
			node* current = stack.pop();

			bool has_visited_parent = false;
			for (node* parent : current->parents) {
				if (parent == start || path.contains(parent)) {
					if (!visited_ancestors.contains(parent))
						visited_ancestors.add(parent);
					has_visited_parent = true;
				}
			}
			if (!has_visited_parent)
				stack.append(current->parents.data, current->parents.length);
		}

		if (current_node->children.length == branch_size && visited_ancestors.contains(current_node))
			visited_ancestors.remove(visited_ancestors.index_of(current_node));
		if (visited_ancestors.length != 0) {
			node* new_parent = choice(visited_ancestors.data, visited_ancestors.length);

			/* replace an existing parent with `new_parent` */
			unsigned int r = randrange(frontier_vertex->parents.length);
			node* old_parent = frontier_vertex->parents[r];
			frontier_vertex->parents[r] = new_parent;
			old_parent->children.remove(old_parent->children.index_of(frontier_vertex));
			new_parent->children.add(frontier_vertex);
			path.add(new_parent);
			path.add(frontier_vertex);
			removable_edges.remove(removable_edges.index_of(make_pair(old_parent->id, frontier_vertex->id)));

			/* move the new edge in `path` to a random position */
			unsigned int insert_path_index = path.index_of(new_parent);
			if (path.length == 0) insert_path_index = 0;
			else if (insert_path_index % 2 == 1) insert_path_index++;
			unsigned int insert_index = randrange(insert_path_index, path.length);
			insert_index -= insert_index % 2;
			for (unsigned int j = path.length - 1; j >= insert_index + 2; j--)
				path[j] = path[j - 2];
			path[insert_index] = new_parent;
			path[insert_index + 1] = frontier_vertex;
		} else {
			/* there is no path from `start` to `frontier_vertex`, so add a new parent that is reachable from `start` */
			array<node*> candidate_parents(8);
			for (node* n : reachable) {
				if (n->id < frontier_vertex->id)
					candidate_parents.add(n);
			}

			if (current_node->children.length == branch_size && candidate_parents.contains(current_node) && candidate_parents.length != 1)
				candidate_parents.remove(candidate_parents.index_of(current_node));

			node* new_parent = choice(candidate_parents.data, candidate_parents.length);
			frontier_vertex->parents.add(new_parent);
			new_parent->children.add(frontier_vertex);
			path.add(new_parent);
			path.add(frontier_vertex);

			/* move the new edge in `path` to a random position */
			unsigned int insert_path_index = path.index_of(new_parent);
			if (path.length == 0) insert_path_index = 0;
			else if (insert_path_index % 2 == 1) insert_path_index++;
			unsigned int insert_index = randrange(insert_path_index, path.length);
			insert_index -= insert_index % 2;
			for (unsigned int j = path.length - 1; j >= insert_index + 2; j--)
				path[j] = path[j - 2];
			path[insert_index] = new_parent;
			path[insert_index + 1] = frontier_vertex;
		}
	}

	/* next we have to make sure every vertex in `frontier` has an unvisited child */
	for (node* frontier_vertex : frontier) {
		bool has_unvisited_child = false;
		for (const node* child : frontier_vertex->children) {
			if (!path.contains(child)) {
				has_unvisited_child = true;
				break;
			}
		}
		if (has_unvisited_child) continue;

		/* randomly select an unvisited vertex after `frontier_vertex` */
		array<unsigned int> unvisited(8);
		for (unsigned int i = frontier_vertex->id + 1; i < vertices.length; i++)
			if (!path.contains(&vertices[i])) unvisited.add(i);
		node* new_child = &vertices[choice(unvisited.data, unvisited.length)];

		/* see if we can replace an unvisited parent node of `new_child` */
		unvisited.clear();
		for (unsigned int i = 0; i < new_child->parents.length; i++)
			if (new_child->parents[i] != start && !path.contains(new_child->parents[i]))
				unvisited.add(i);
		if (unvisited.length != 0) {
			unsigned int r = choice(unvisited.data, unvisited.length);
			node* old_parent = new_child->parents[r];
			old_parent->children.remove(old_parent->children.index_of(new_child));
			frontier_vertex->children.add(new_child);
			new_child->parents[r] = frontier_vertex;
			removable_edges.remove(removable_edges.index_of(make_pair(old_parent->id, new_child->id)));
		} else {
			/* we can't remove any parents from `new_child` so simply add an edge */
			frontier_vertex->children.add(new_child);
			new_child->parents.add(frontier_vertex);
			total_edge_count++; new_edge_count++;
		}
	}

	/* next, we want to make sure `vertices[current_node_index]` has an edge to `branch_size` vertices that follows it */
	unsigned int current_branch_size = 0;
	available_nodes.clear();
	array<node*> available_frontier_branches(8);
	for (unsigned int i = current_node_index + 1; i < vertices.length; i++) {
		available_nodes.add(&vertices[i]);

		if (frontier.contains(&vertices[i]))
			available_frontier_branches.add(&vertices[i]);
	}
	unsigned int num_frontier_branches = 0;
	while (available_frontier_branches.length > 0 && num_frontier_branches < max(2*frontier_size + branch_size, max_edges) - max_edges) {
		unsigned int r = randrange(available_frontier_branches.length);
		node* new_child = available_frontier_branches[r];
		available_frontier_branches.remove(r);
		available_nodes.remove(available_nodes.index_of(new_child));

		/* find the edge in the path that leads to `new_child` */
		unsigned int path_index;
		for (path_index = 0; path_index < path.length; path_index += 2)
			if (path[path_index + 1] == new_child) break;

		if (path[path_index] == current_node) {
			/* this vertex is already visited from current */
			unsigned int index = removable_edges.index_of(make_pair(current_node_index, new_child->id));
			if (index < removable_edges.length)
				removable_edges.remove(index);
			current_branch_size++;
			num_frontier_branches++;
			continue;
		}

		/* connect this vertex directly to `current_node` */
		array<unsigned int> visited(8);
		node* old_parent = (node*) path[path_index];
		old_parent->children.remove(old_parent->children.index_of(new_child));
		if (current_node->children.contains(new_child)) {
			new_child->parents.remove(new_child->parents.index_of(old_parent));
		} else {
			current_node->children.add(new_child);
			new_child->parents[new_child->parents.index_of(old_parent)] = current_node;
		}
		unsigned int index = removable_edges.index_of(make_pair(old_parent->id, new_child->id));
		if (index < removable_edges.length)
			removable_edges.remove(index);
		index = removable_edges.index_of(make_pair(current_node_index, new_child->id));
		if (index < removable_edges.length)
			removable_edges.remove(index);

		/* replace the corresponding edge from `path` */
		path[path_index] = current_node;

		current_branch_size++;
		num_frontier_branches++;
	}
	while (current_branch_size < branch_size) {
		unsigned int r = randrange(available_nodes.length);
		node* new_child = available_nodes[r];
		available_nodes.remove(r);

		if (current_node->children.contains(new_child)){
			unsigned int index = removable_edges.index_of(make_pair(current_node_index, new_child->id));
			if (index < removable_edges.length)
				removable_edges.remove(index);
			current_branch_size++;
			continue;
		}

		/* see if we can replace an unvisited parent node of `new_child` */
		array<unsigned int> unvisited(8);
		for (unsigned int i = 0; i < new_child->parents.length; i++)
			if (new_child->parents[i] != start && !path.contains(new_child->parents[i]))
				unvisited.add(i);
		if (unvisited.length != 0) {
			unsigned int r = choice(unvisited.data, unvisited.length);
			node* old_parent = new_child->parents[r];
			old_parent->children.remove(old_parent->children.index_of(new_child));
			current_node->children.add(new_child);
			new_child->parents[r] = current_node;
			removable_edges.remove(removable_edges.index_of(make_pair(old_parent->id, new_child->id)));
		} else {
			/* we can't remove any parents from `new_child` so simply add an edge */
			current_node->children.add(new_child);
			new_child->parents.add(current_node);
			total_edge_count++; new_edge_count++;
		}
		current_branch_size++;
	}

	/* visit some of the children of `current_node` */
	//unsigned int visited_branches = randrange(branch_size);
	//array<node*> available_nodes(8);
	//available_nodes.append(current_node->children.data, current_node->children.length);
	//unsigned int index = available_nodes.index_of(end);
	//if (index != available_nodes.length)
	//	available_nodes.remove(index); /* make sure we don't visit the goal vertex prematurely */
	//unsigned int insert_path_index = path.index_of(current_node) + 1;
	//if (path.length == 0) insert_path_index = 0;
	//else if (insert_path_index % 2 == 1) insert_path_index++;
	//for (unsigned int i = 0; i < visited_branches && available_nodes.length != 0; i++) {
		/* count the number of branches with unvisited children */
	//	unsigned int r = randrange(available_nodes.length);
	//	node* child = available_nodes[r];
	//	available_nodes.remove(r);

	//	path.add(current_node);
	//	path.add(child);
	//	unsigned int insert_index = randrange(insert_path_index, path.length);
	//	insert_index -= insert_index % 2;
	//	for (unsigned int j = path.length - 1; j >= insert_index + 2; j--)
	//		path[j] = path[j - 2];
	//	path[insert_index] = current_node;
	//	path[insert_index + 1] = child;
	//}

	/* make sure `end` is reachable from `start` */
	array<pair<node*, const node*>> reachability_stack(8);
	for (node* child : start->children)
		reachability_stack.add(make_pair(child, start));
	array_map<node*, const node*> reverse_ptrs(8);
	while (reachability_stack.length != 0) {
		pair<node*, const node*> entry = reachability_stack.pop();
		node* next = entry.key;
		const node* parent = entry.value;
		if (next->id > end_index || reverse_ptrs.contains(next))
			continue;
		reverse_ptrs.put(next, parent);
		for (node* child : next->children)
			if (!reachability_stack.contains(make_pair<node*, const node*>(child, next)))
				reachability_stack.add(make_pair<node*, const node*>(child, next));
	}
	const node* current;
	const node* parent;
	if (reverse_ptrs.contains(&vertices[end_index])) {
		current = &vertices[end_index];
		parent = reverse_ptrs.get(current);
	} else {
		reverse_ptrs.put(&vertices[start_index], nullptr);
		node* new_parent = choice(reverse_ptrs.keys, reverse_ptrs.size);
		new_parent->children.add(&vertices[end_index]);
		vertices[end_index].parents.add(new_parent);
		total_edge_count++; new_edge_count++;

		current = &vertices[end_index];
		parent = new_parent;
	}
	/* make sure none of the edges on the path from `start` to `end` are removable */
	while (true) {
		pair<unsigned int, unsigned int> entry = make_pair(parent->id, current->id);
		unsigned int index = removable_edges.index_of(entry);
		if (index != removable_edges.length)
			removable_edges.remove(index);
		if (parent == start)
			break;
		current = parent;
		parent = reverse_ptrs.get(current);
	}

	/* remove edges to avoid generating a graph with too many edges */
	unsigned int edges_to_remove;
	if (total_edge_count > max_edges)
		edges_to_remove = total_edge_count - max_edges;
	else
		edges_to_remove = new_edge_count;
	for (unsigned int i = 0; i < edges_to_remove && removable_edges.length != 0; i++) {
		unsigned int u = randrange(removable_edges.length);
		unsigned int parent_id = removable_edges[u].key;
		unsigned int child_id = removable_edges[u].value;
		removable_edges.remove(u);

		/* make sure that removing this edge will not remove an unvisited child node from any frontier vertex */
		if (frontier.contains(&vertices[parent_id])) {
			bool can_remove_edge = false;
			for (const node* child : vertices[parent_id].children) {
				if (child->id != child_id && !path.contains(child)) {
					can_remove_edge = true;
					break;
				}
			}
			if (!can_remove_edge) {
				i--;
				continue;
			}
		}

		vertices[parent_id].children.remove(vertices[parent_id].children.index_of(&vertices[child_id]));
		vertices[child_id].parents.remove(vertices[child_id].parents.index_of(&vertices[parent_id]));
	}

	/* visit some additional vertices */
	array<float> path_length_probabilities(8);
	for (unsigned int i = path.length / 2; i < max_edges + 1; i++)
		path_length_probabilities.add((float) i);
	unsigned int target_path_length = path.length / 2 + sample_categorical(path_length_probabilities.data, path_length_probabilities.length);
	available_nodes.clear();
	for (node* frontier_vertex : frontier) {
		unsigned int unvisited_child_count = 0;
		for (const node* child : frontier_vertex->children)
			if (!path.contains(child)) unvisited_child_count++;

		if (unvisited_child_count > 1)
			available_nodes.add(frontier_vertex);
	}
	while (available_nodes.length != 0 && path.length < target_path_length * 2) {
		unsigned int u = randrange(available_nodes.length);
		node* selected_vertex = available_nodes[u];
		available_nodes.remove(u);

		array<node*> unvisited_children(4);
		for (node* child : selected_vertex->children) {
			if (path.contains(child)) continue;

			/* make sure this is not the unvisited child of another frontier vertex */
			bool can_visit = true;
			for (const node* parent : child->parents) {
				if (!frontier.contains(parent)) continue;
				unsigned int unvisited_child_count = 0;
				for (const node* other_child : parent->children)
					if (!path.contains(other_child)) unvisited_child_count++;
				if (unvisited_child_count == 1) {
					can_visit = false;
					break;
				}
			}
			if (can_visit)
				unvisited_children.add(child);
		}
		if (unvisited_children.length == 0)
			continue;
		u = randrange(unvisited_children.length);
		node* selected_child = unvisited_children[u];
		path.add(selected_vertex);
		path.add(selected_child);
		if (frontier.contains(selected_vertex) && unvisited_children.length - 1 > 1)
			available_nodes.add(selected_vertex);
		else if (!frontier.contains(selected_vertex) && unvisited_children.length - 1 > 0)
			available_nodes.add(selected_vertex);

		/* check if `selected_child` has any unvisited child nodes */
		unsigned int unvisited_child_count = 0;
		for (const node* child : selected_child->children)
			if (!path.contains(child)) unvisited_child_count++;
		if (unvisited_child_count > 0)
			available_nodes.add(selected_child);
	}

	/* remove any correlation between graph topology and vertex IDs by shuffling the vertices */
	unsigned int* new_indices = (unsigned int*) alloca(sizeof(unsigned int) * (max_vertex_id + 1));
	for (unsigned int i = 0; i < max_vertex_id + 1; i++) new_indices[i] = i;
	shuffle(new_indices, max_vertex_id + 1);
	unsigned int src_index = 0;
	for (unsigned int i = 0; i < vertices.length; i++) {
		bool is_reserved = false;
		for (unsigned int j = 0; j < array_length(RESERVED_INDICES); j++) {
			if (new_indices[src_index] == RESERVED_INDICES[j]) {
				is_reserved = true;
				break;
			}
		}
		if (is_reserved)
			src_index++;
		vertices[i].id = new_indices[src_index];
		src_index++;
	}

	return true;
}

py::tuple generate_si_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const py::object& reserved_inputs, const int requested_frontier_size, const int requested_branch_size, bool uniform, const bool quiet=false)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int max_edges = (max_input_size - 2) / 6;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	unsigned int ntokens = (max_input_size - 5) / 3 + 5;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, ntokens};
	size_t label_shape[1]{dataset_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	py::array_t<int64_t, py::array::c_style> labels(label_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	auto labels_mem = labels.mutable_unchecked<1>();

	unsigned int* frontier_branch_histogram = (unsigned int*) alloca(sizeof(unsigned int) * (max_edges + 1) * (max_edges + 1));
	for (unsigned int i = 0; i < (max_edges + 1) * (max_edges + 1); i++)
		frontier_branch_histogram[i] = 0;
	unsigned int* visited_edges_histogram = (unsigned int*) alloca(sizeof(unsigned int) * (max_edges + 1));
	for (unsigned int i = 0; i < max_edges + 1; i++)
		visited_edges_histogram[i] = 0;

	float* MAX_FREQS_PER_BUCKET = (float*) alloca(sizeof(float) * (max_edges + 1) * (max_edges + 1));
	unsigned int nonzero_buckets = 0;
	for (unsigned int i = 0; i < max_edges + 1; i++) {
		/* frontier_size is i */
		for (unsigned int j = 0; j < max_edges + 1; j++) {
			/* branch_size is j */
			if (i == 0 || j == 0 || i > (unsigned) requested_frontier_size || j > (unsigned) requested_branch_size || i + j > max_edges + 1) {
				MAX_FREQS_PER_BUCKET[i*(max_edges+1) + j] = 0.0;
			} else {
				MAX_FREQS_PER_BUCKET[i*(max_edges+1) + j] = 1.0;
				nonzero_buckets++;
			}
		}
	}
	for (unsigned int i = 0; i < (max_edges + 1) * (max_edges + 1); i++)
		if (MAX_FREQS_PER_BUCKET[i] != 0.0) MAX_FREQS_PER_BUCKET[i] /= nonzero_buckets;
	if ((unsigned) requested_branch_size < max_edges + 1 - requested_frontier_size)
		MAX_FREQS_PER_BUCKET[requested_frontier_size*(max_edges+1) + requested_branch_size] += 0.05 / nonzero_buckets;
	else
		MAX_FREQS_PER_BUCKET[requested_frontier_size*(max_edges+1) + max_edges + 1 - requested_frontier_size] += 0.05 / nonzero_buckets;

	array<const node*> path(32);
	pair<unsigned int, unsigned int>* potential_frontier_branches = (pair<unsigned int, unsigned int>*) alloca(max((size_t) 1, sizeof(pair<unsigned int, unsigned int>) * ((max_edges + 1) * (max_edges + 1))));
	unsigned int potential_frontier_branch_count = 0;
	unsigned int num_attempts = 0;
	while (num_generated < dataset_size) {
		if (num_attempts >= 10000000)
			break;
		num_attempts++;
		array<node> g(32);
		const node* start; const node* end;
		unsigned int current_node_index;
		while (true) {
			unsigned int frontier_size; unsigned int branch_size;
			if (uniform) {
				potential_frontier_branch_count = 0;
				for (unsigned int i = 1; i < max_edges + 1; i++) {
					for (unsigned int j = 1; j < max_edges + 1; j++) {
						if ((num_generated == 0 && MAX_FREQS_PER_BUCKET[i*(max_edges+1) + j] != 0.0) || (float) frontier_branch_histogram[i*(max_edges+1) + j] / num_generated < MAX_FREQS_PER_BUCKET[i*(max_edges+1) + j])
							potential_frontier_branches[potential_frontier_branch_count++] = make_pair(i, j);
					}
				}
				pair<unsigned int, unsigned int> frontier_branch = choice(potential_frontier_branches, potential_frontier_branch_count);
				frontier_size = frontier_branch.key;
				branch_size = frontier_branch.value;
			} else {
				frontier_size = requested_frontier_size;
				branch_size = requested_branch_size;
			}
			unsigned int num_vertices = std::max(2u, randrange(max_edges - frontier_size + 1));
			num_vertices = std::max(frontier_size + branch_size + 1 - std::min(frontier_size, branch_size), num_vertices);
			if (!generate_si_example(g, start, end, current_node_index, path, num_vertices, max_input_size / 24 + 1, (max_input_size - 2) / 6 + 2, max_edges, frontier_size, branch_size)) {
				for (node& n : g) core::free(n);
				g.length = 0; path.length = 0;
				continue;
			}
			break;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges) {
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		/* compute the frontier size and branch size of this example */
		array<const node*> frontier(8);
		for (unsigned int i = 0; i < path.length; i++) {
			bool has_unvisited_children = false;
			for (const node* child : path[i]->children) {
				if (!path.contains(child)) {
					has_unvisited_children = true;
					break;
				}
			}
			if (has_unvisited_children && !frontier.contains(path[i]))
				frontier.add(path[i]);
		}
		if (frontier.length == 0)
			frontier.add(start);
		const node* current_node = &g[current_node_index];
		unsigned int branch_size = current_node->children.length;

		bool is_selection_step;
		if (path.length == 0)
			is_selection_step = false;
		else
			is_selection_step = (randrange(2) == 0);
		if (3*(path.length/2) + (is_selection_step ? 1 : 2) > 3*(max_edges - 1) + 1) {
			/* we have just barely too many edges */
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			continue;
		}
		for (unsigned int j = 3*(path.length/2) + (is_selection_step ? 1 : 2); j < 3*(max_edges - 1) + 1; j++)
			prefix[prefix.length++] = PATH_PREFIX_TOKEN;
		for (unsigned int j = 0; j < path.length; j += 2) {
			prefix[prefix.length++] = PATH_PREFIX_TOKEN;
			prefix[prefix.length++] = path[j]->id;
			prefix[prefix.length++] = path[j+1]->id;
		}
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;
		if (!is_selection_step)
			prefix[prefix.length++] = current_node->id;

		if ((requested_frontier_size != -1 && !uniform && (unsigned int) requested_frontier_size != (unsigned int) frontier.length)
		 || (requested_branch_size != -1 && !uniform && (unsigned int) requested_branch_size != branch_size)
		 || (uniform && num_generated != 0 && (float) frontier_branch_histogram[frontier.length*(max_edges+1) + branch_size] / num_generated >= MAX_FREQS_PER_BUCKET[frontier.length*(max_edges+1) + branch_size]))
		{
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			continue;
		}

		/* check if this input is reserved */
		py::object contains = reserved_inputs.attr("__contains__");
		py::tuple example_tuple(prefix.length);
		for (unsigned int i = 0; i < prefix.length; i++)
			example_tuple[i] = prefix[i];
		if (contains(example_tuple).is(py_true)) {
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			num_collisions += 1;
			continue;
		}

		frontier_branch_histogram[frontier.length*(max_edges+1) + branch_size]++;
		visited_edges_histogram[path.length/2]++;

		for (unsigned int i = 0; i < max_input_size - prefix.length; i++)
			inputs_mem(num_generated, i) = PADDING_TOKEN;
		for (unsigned int i = 0; i < prefix.length; i++)
			inputs_mem(num_generated, max_input_size - prefix.length + i) = prefix[i];
		for (unsigned int i = 0; i < ntokens; i++)
			outputs_mem(num_generated, i) = 0.0f;
		if (is_selection_step) {
			for (const node* frontier_vertex : frontier)
				outputs_mem(num_generated, frontier_vertex->id) = 1.0f;
			labels_mem(num_generated) = choice(frontier.data, frontier.length)->id;
		} else {
			array<unsigned int> correct_answers(8);
			for (const node* child : current_node->children) {
				if (!path.contains(child)) {
					outputs_mem(num_generated, child->id) = 1.0f;
					correct_answers.add(child->id);
				}
			}
			labels_mem(num_generated) = choice(correct_answers.data, correct_answers.length);
		}
		num_generated++;

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= dataset_size)) {
			printf("%d examples generated.\n", num_generated);
			fflush(stdout);

			printf("Frontier-branch size histogram: (log frequencies)\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_edges + 1; i++) {
				for (unsigned int j = 0; j < max_edges + 1; j++) {
					if (frontier_branch_histogram[i*(max_edges+1) + j] == 0)
						continue;
					if (!first) printf(", ");
					printf("(%d,%d):%.2f", i, j, log(frontier_branch_histogram[i*(max_edges+1) + j]) - log(num_generated));
					first = false;
				}
			}
			printf("]\n");

			printf("Visited edge count histogram: (log frequencies)\n");
			printf("[");
			first = true;
			for (unsigned int i = 0; i < max_edges + 1; i++) {
				if (visited_edges_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, log(visited_edges_histogram[i]) - log(num_generated));
				first = false;
			}
			printf("]\n");
		}

		for (node& n : g) core::free(n);
		g.length = 0; path.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, labels, num_collisions);
}

PYBIND11_MODULE(generator, m) {
	m.def("generate_training_set", &generate_training_set);
	m.def("generate_reachable_training_set", &generate_reachable_training_set);
	m.def("generate_dfs_training_set", &generate_dfs_training_set);
	m.def("generate_si_training_set", &generate_si_training_set);
	m.def("lookahead_histogram", &lookahead_histogram);
	m.def("set_seed", &core::set_seed);
}
