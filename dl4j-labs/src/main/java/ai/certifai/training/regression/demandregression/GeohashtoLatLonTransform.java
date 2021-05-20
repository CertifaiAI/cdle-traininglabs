/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.certifai.training.regression.demandregression;


import com.github.davidmoten.geo.GeoHash;
import com.github.davidmoten.geo.LatLong;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import static ai.certifai.training.regression.demandregression.CoordinatesType.LAT;
import static ai.certifai.training.regression.demandregression.CoordinatesType.LON;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public class GeohashtoLatLonTransform implements Transform {

    private final String columnName;
    private final String insertAfter;
    private final List<DerivedColumn> derivedColumns;
    private Schema inputSchema;
    private int insertAfterIdx = -1;
    private int deriveFromIdx = -1;


    private GeohashtoLatLonTransform(Builder builder) {
        this.derivedColumns = builder.derivedColumns;
        this.columnName = builder.columnName;
        this.insertAfter = builder.insertAfter;
    }

    public GeohashtoLatLonTransform(@JsonProperty("columnName") String columnName,
                                    @JsonProperty("insertAfter") String insertAfter,
                                    @JsonProperty("derivedColumns") List<DerivedColumn> derivedColumns) {
        this.columnName = columnName;
        this.insertAfter = insertAfter;
        this.derivedColumns = derivedColumns;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        List<ColumnMetaData> oldMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(oldMeta.size() + derivedColumns.size());

        List<String> oldNames = inputSchema.getColumnNames();

        for (int i = 0; i < oldMeta.size(); i++) {
            String current = oldNames.get(i);
            newMeta.add(oldMeta.get(i));

            if (insertAfter.equals(current)) {
                //Insert the derived columns here
                for (DerivedColumn d : derivedColumns) {
                    newMeta.add(new DoubleMetaData(d.columnName));
                }
            }
        }

        return inputSchema.newSchema(newMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        insertAfterIdx = inputSchema.getColumnNames().indexOf(insertAfter);
        if (insertAfterIdx == -1) {
            throw new IllegalStateException(
                    "Invalid schema/insert after column: input schema does not contain column \"" + insertAfter
                            + "\"");
        }

        deriveFromIdx = inputSchema.getColumnNames().indexOf(columnName);
        if (deriveFromIdx == -1) {
            throw new IllegalStateException(
                    "Invalid source column: input schema does not contain column \"" + columnName + "\"");
        }

        this.inputSchema = inputSchema;

    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                    + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                    + "). Transform = " + toString());
        }

        int i = 0;
        Writable source = writables.get(deriveFromIdx);
        LatLong latlong = GeoHash.decodeHash(source.toString());
        List<Writable> list = new ArrayList<>(writables.size() + derivedColumns.size());
        for (Writable w : writables) {
            list.add(w);
            if (i++ == insertAfterIdx) {
                for (DerivedColumn d : derivedColumns) {
                    switch (d.coordinatesType) {
                        case LAT:
                            list.add(new DoubleWritable(latlong.getLat()));
                            break;
                        case LON:
                            list.add(new DoubleWritable(latlong.getLon()));
                            break;
                        default:
                            throw new IllegalStateException("Unexpected column type: " + d.columnType);
                    }
                }
            }
        }
        return list;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>(sequence.size());
        for (List<Writable> step : sequence) {
            out.add(map(step));
        }
        return out;
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        List<Object> ret = new ArrayList<>();
        String geohashstr = (String) input;
        LatLong latlong = GeoHash.decodeHash(geohashstr);
        for (DerivedColumn d : derivedColumns) {
            switch (d.coordinatesType) {
                case LAT:
                    ret.add(latlong.getLat());
                    break;

                case LON:
                    ret.add(latlong.getLon());
                    break;
                default:
                    throw new IllegalStateException("Unexpected column type: " + d.coordinatesType);
            }
        }

        return ret;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<Long> longs = (List<Long>) sequence;
        List<List<Object>> ret = new ArrayList<>();
        for (Long l : longs)
            ret.add((List<Object>) map(l));
        return ret;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("GeohashtoLatLonTransform(\"").append(columnName).append("\",insertAfter=\"")
                .append(insertAfter).append("\",derivedColumns=(");

        boolean first = true;
        for (DerivedColumn d : derivedColumns) {
            if (!first)
                sb.append(",");
            sb.append(d);
            first = false;
        }

        sb.append("))");

        return sb.toString();
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return outputColumnNames()[0];
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        String[] ret = new String[derivedColumns.size()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = derivedColumns.get(i).columnName;
        return ret;
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return new String[] {columnName()};
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnName;
    }

    public static class Builder {

        private final String columnName;
        private String insertAfter;
        private final List<DerivedColumn> derivedColumns = new ArrayList<>();


        /**
         * @param geohashColumnName The name of the geohash column from which to derive the new values
         */
        public Builder(String geohashColumnName) {
            this.columnName = geohashColumnName;
            this.insertAfter = geohashColumnName;
        }


        public Builder insertAfter(String columnName) {
            this.insertAfter = columnName;
            return this;
        }


        public Builder addLatDerivedColumn(String columnName ) {
            derivedColumns.add(new DerivedColumn(columnName, ColumnType.Double, LAT));
            return this;
        }

        public Builder addLonDerivedColumn(String columnName ) {
            derivedColumns.add(new DerivedColumn(columnName, ColumnType.Double, LON));
            return this;
        }

        /**
         * Create the transform instance
         */
        public GeohashtoLatLonTransform build() {
            return new GeohashtoLatLonTransform(this);
        }
    }


    public static class DerivedColumn implements Serializable {
        private final String columnName;
        private final ColumnType columnType;
        private final CoordinatesType coordinatesType; //o0o


        public DerivedColumn(@JsonProperty("columnName") String columnName,
                             @JsonProperty("columnType") ColumnType columnType,
                             @JsonProperty("coordinatesType") CoordinatesType coordinatesType
        ) {
            this.columnName = columnName;
            this.columnType = columnType;
            this.coordinatesType = coordinatesType;
        }

        @Override
        public String toString() {
            return "(name=" + columnName + ",type=" + columnType + ",derived=" + ")"; //TODO: to update
        }

    }

}
