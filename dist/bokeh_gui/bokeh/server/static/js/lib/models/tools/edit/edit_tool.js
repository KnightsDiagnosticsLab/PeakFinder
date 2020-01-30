"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const p = require("../../../core/properties");
const array_1 = require("../../../core/util/array");
const types_1 = require("../../../core/util/types");
const gesture_tool_1 = require("../gestures/gesture_tool");
class EditToolView extends gesture_tool_1.GestureToolView {
    constructor() {
        super(...arguments);
        this._mouse_in_frame = true;
    }
    _move_enter(_e) {
        this._mouse_in_frame = true;
    }
    _move_exit(_e) {
        this._mouse_in_frame = false;
    }
    _map_drag(sx, sy, renderer) {
        // Maps screen to data coordinates
        const frame = this.plot_view.frame;
        if (!frame.bbox.contains(sx, sy)) {
            return null;
        }
        const x = frame.xscales[renderer.x_range_name].invert(sx);
        const y = frame.yscales[renderer.y_range_name].invert(sy);
        return [x, y];
    }
    _delete_selected(renderer) {
        // Deletes all selected rows in the ColumnDataSource
        const cds = renderer.data_source;
        const indices = cds.selected.indices;
        indices.sort();
        for (const column of cds.columns()) {
            const values = cds.get_array(column);
            for (let index = 0; index < indices.length; index++) {
                const ind = indices[index];
                values.splice(ind - index, 1);
            }
        }
        this._emit_cds_changes(cds);
    }
    _pop_glyphs(cds, num_objects) {
        // Pops rows in the CDS until only num_objects are left
        const columns = cds.columns();
        if (!num_objects || !columns.length)
            return;
        for (const column of columns) {
            let array = cds.get_array(column);
            const drop = array.length - num_objects + 1;
            if (drop < 1)
                continue;
            if (!types_1.isArray(array)) {
                array = Array.from(array);
                cds.data[column] = array;
            }
            array.splice(0, drop);
        }
    }
    _emit_cds_changes(cds, redraw = true, clear = true, emit = true) {
        if (clear)
            cds.selection_manager.clear();
        if (redraw)
            cds.change.emit();
        if (emit) {
            cds.data = cds.data;
            cds.properties.data.change.emit();
        }
    }
    _drag_points(ev, renderers) {
        if (this._basepoint == null)
            return;
        const [bx, by] = this._basepoint;
        for (const renderer of renderers) {
            const basepoint = this._map_drag(bx, by, renderer);
            const point = this._map_drag(ev.sx, ev.sy, renderer);
            if (point == null || basepoint == null) {
                continue;
            }
            const [x, y] = point;
            const [px, py] = basepoint;
            const [dx, dy] = [x - px, y - py];
            // Type once dataspecs are typed
            const glyph = renderer.glyph;
            const cds = renderer.data_source;
            const [xkey, ykey] = [glyph.x.field, glyph.y.field];
            for (const index of cds.selected.indices) {
                if (xkey)
                    cds.data[xkey][index] += dx;
                if (ykey)
                    cds.data[ykey][index] += dy;
            }
            cds.change.emit();
        }
        this._basepoint = [ev.sx, ev.sy];
    }
    _pad_empty_columns(cds, coord_columns) {
        // Pad ColumnDataSource non-coordinate columns with empty_value
        for (const column of cds.columns()) {
            if (!array_1.includes(coord_columns, column))
                cds.get_array(column).push(this.model.empty_value);
        }
    }
    _select_event(ev, append, renderers) {
        // Process selection event on the supplied renderers and return selected renderers
        const frame = this.plot_view.frame;
        const { sx, sy } = ev;
        if (!frame.bbox.contains(sx, sy)) {
            return [];
        }
        const geometry = { type: 'point', sx, sy };
        const selected = [];
        for (const renderer of renderers) {
            const sm = renderer.get_selection_manager();
            const cds = renderer.data_source;
            const views = [this.plot_view.renderer_views[renderer.id]];
            const did_hit = sm.select(views, geometry, true, append);
            if (did_hit) {
                selected.push(renderer);
            }
            cds.properties.selected.change.emit();
        }
        return selected;
    }
}
exports.EditToolView = EditToolView;
EditToolView.__name__ = "EditToolView";
class EditTool extends gesture_tool_1.GestureTool {
    constructor(attrs) {
        super(attrs);
    }
    static init_EditTool() {
        this.define({
            custom_icon: [p.String],
            custom_tooltip: [p.String],
            empty_value: [p.Any],
            renderers: [p.Array, []],
        });
    }
    get tooltip() {
        return this.custom_tooltip || this.tool_name;
    }
    get computed_icon() {
        return this.custom_icon || this.icon;
    }
}
exports.EditTool = EditTool;
EditTool.__name__ = "EditTool";
EditTool.init_EditTool();
