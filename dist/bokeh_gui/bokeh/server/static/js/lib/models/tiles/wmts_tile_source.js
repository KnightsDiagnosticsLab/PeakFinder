"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mercator_tile_source_1 = require("./mercator_tile_source");
class WMTSTileSource extends mercator_tile_source_1.MercatorTileSource {
    constructor(attrs) {
        super(attrs);
    }
    get_image_url(x, y, z) {
        const image_url = this.string_lookup_replace(this.url, this.extra_url_vars);
        const [wx, wy, wz] = this.tms_to_wmts(x, y, z);
        return image_url
            .replace("{X}", wx.toString())
            .replace("{Y}", wy.toString())
            .replace("{Z}", wz.toString());
    }
}
exports.WMTSTileSource = WMTSTileSource;
WMTSTileSource.__name__ = "WMTSTileSource";
