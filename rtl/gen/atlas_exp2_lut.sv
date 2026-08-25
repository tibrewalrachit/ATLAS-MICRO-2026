//===========================================================================
// atlas_exp2_lut -- GENERATED FILE, do not edit
//
// Regenerate with:  python3 tb/golden/gen_exp2_lut.py rtl/gen/atlas_exp2_lut.sv
//
// Piecewise-linear table for 2^f, f in [0,1):
//
//     2^f  ~  (base + ((delta * frac) >> 16)) * 2^-23
//
// base lands in [2^23, 2^24), so its low 23 bits are directly the binary32
// significand of the result.  Bases are pre-shifted by half a segment's peak
// interpolation error, which centres the approximation on the true curve
// rather than leaving it uniformly low.
//
// 256 segments; worst-case relative error is reported by the generator.
//===========================================================================

module atlas_exp2_lut (
  input  logic [7:0]  idx,
  output logic [23:0]  base,
  output logic [15:0]  delta
);

  always_comb begin
    case (idx)
      8'd0   : begin base = 24'd8388616 ; delta = 16'd22744 ; end
      8'd1   : begin base = 24'd8411360 ; delta = 16'd22805 ; end
      8'd2   : begin base = 24'd8434165 ; delta = 16'd22867 ; end
      8'd3   : begin base = 24'd8457032 ; delta = 16'd22929 ; end
      8'd4   : begin base = 24'd8479962 ; delta = 16'd22991 ; end
      8'd5   : begin base = 24'd8502953 ; delta = 16'd23054 ; end
      8'd6   : begin base = 24'd8526007 ; delta = 16'd23116 ; end
      8'd7   : begin base = 24'd8549123 ; delta = 16'd23179 ; end
      8'd8   : begin base = 24'd8572302 ; delta = 16'd23242 ; end
      8'd9   : begin base = 24'd8595544 ; delta = 16'd23305 ; end
      8'd10  : begin base = 24'd8618849 ; delta = 16'd23368 ; end
      8'd11  : begin base = 24'd8642217 ; delta = 16'd23431 ; end
      8'd12  : begin base = 24'd8665649 ; delta = 16'd23495 ; end
      8'd13  : begin base = 24'd8689144 ; delta = 16'd23559 ; end
      8'd14  : begin base = 24'd8712702 ; delta = 16'd23623 ; end
      8'd15  : begin base = 24'd8736325 ; delta = 16'd23687 ; end
      8'd16  : begin base = 24'd8760011 ; delta = 16'd23751 ; end
      8'd17  : begin base = 24'd8783762 ; delta = 16'd23815 ; end
      8'd18  : begin base = 24'd8807577 ; delta = 16'd23880 ; end
      8'd19  : begin base = 24'd8831457 ; delta = 16'd23944 ; end
      8'd20  : begin base = 24'd8855402 ; delta = 16'd24009 ; end
      8'd21  : begin base = 24'd8879411 ; delta = 16'd24075 ; end
      8'd22  : begin base = 24'd8903486 ; delta = 16'd24140 ; end
      8'd23  : begin base = 24'd8927625 ; delta = 16'd24205 ; end
      8'd24  : begin base = 24'd8951831 ; delta = 16'd24271 ; end
      8'd25  : begin base = 24'd8976102 ; delta = 16'd24337 ; end
      8'd26  : begin base = 24'd9000438 ; delta = 16'd24403 ; end
      8'd27  : begin base = 24'd9024841 ; delta = 16'd24469 ; end
      8'd28  : begin base = 24'd9049310 ; delta = 16'd24535 ; end
      8'd29  : begin base = 24'd9073845 ; delta = 16'd24602 ; end
      8'd30  : begin base = 24'd9098447 ; delta = 16'd24668 ; end
      8'd31  : begin base = 24'd9123115 ; delta = 16'd24735 ; end
      8'd32  : begin base = 24'd9147850 ; delta = 16'd24802 ; end
      8'd33  : begin base = 24'd9172653 ; delta = 16'd24870 ; end
      8'd34  : begin base = 24'd9197522 ; delta = 16'd24937 ; end
      8'd35  : begin base = 24'd9222459 ; delta = 16'd25005 ; end
      8'd36  : begin base = 24'd9247464 ; delta = 16'd25072 ; end
      8'd37  : begin base = 24'd9272536 ; delta = 16'd25140 ; end
      8'd38  : begin base = 24'd9297677 ; delta = 16'd25209 ; end
      8'd39  : begin base = 24'd9322885 ; delta = 16'd25277 ; end
      8'd40  : begin base = 24'd9348162 ; delta = 16'd25345 ; end
      8'd41  : begin base = 24'd9373508 ; delta = 16'd25414 ; end
      8'd42  : begin base = 24'd9398922 ; delta = 16'd25483 ; end
      8'd43  : begin base = 24'd9424405 ; delta = 16'd25552 ; end
      8'd44  : begin base = 24'd9449957 ; delta = 16'd25621 ; end
      8'd45  : begin base = 24'd9475578 ; delta = 16'd25691 ; end
      8'd46  : begin base = 24'd9501269 ; delta = 16'd25761 ; end
      8'd47  : begin base = 24'd9527030 ; delta = 16'd25830 ; end
      8'd48  : begin base = 24'd9552860 ; delta = 16'd25900 ; end
      8'd49  : begin base = 24'd9578761 ; delta = 16'd25971 ; end
      8'd50  : begin base = 24'd9604731 ; delta = 16'd26041 ; end
      8'd51  : begin base = 24'd9630772 ; delta = 16'd26112 ; end
      8'd52  : begin base = 24'd9656884 ; delta = 16'd26182 ; end
      8'd53  : begin base = 24'd9683067 ; delta = 16'd26253 ; end
      8'd54  : begin base = 24'd9709320 ; delta = 16'd26325 ; end
      8'd55  : begin base = 24'd9735645 ; delta = 16'd26396 ; end
      8'd56  : begin base = 24'd9762041 ; delta = 16'd26468 ; end
      8'd57  : begin base = 24'd9788508 ; delta = 16'd26539 ; end
      8'd58  : begin base = 24'd9815048 ; delta = 16'd26611 ; end
      8'd59  : begin base = 24'd9841659 ; delta = 16'd26683 ; end
      8'd60  : begin base = 24'd9868342 ; delta = 16'd26756 ; end
      8'd61  : begin base = 24'd9895098 ; delta = 16'd26828 ; end
      8'd62  : begin base = 24'd9921926 ; delta = 16'd26901 ; end
      8'd63  : begin base = 24'd9948827 ; delta = 16'd26974 ; end
      8'd64  : begin base = 24'd9975801 ; delta = 16'd27047 ; end
      8'd65  : begin base = 24'd10002849; delta = 16'd27120 ; end
      8'd66  : begin base = 24'd10029969; delta = 16'd27194 ; end
      8'd67  : begin base = 24'd10057163; delta = 16'd27268 ; end
      8'd68  : begin base = 24'd10084431; delta = 16'd27342 ; end
      8'd69  : begin base = 24'd10111772; delta = 16'd27416 ; end
      8'd70  : begin base = 24'd10139188; delta = 16'd27490 ; end
      8'd71  : begin base = 24'd10166678; delta = 16'd27565 ; end
      8'd72  : begin base = 24'd10194243; delta = 16'd27639 ; end
      8'd73  : begin base = 24'd10221882; delta = 16'd27714 ; end
      8'd74  : begin base = 24'd10249597; delta = 16'd27789 ; end
      8'd75  : begin base = 24'd10277386; delta = 16'd27865 ; end
      8'd76  : begin base = 24'd10305251; delta = 16'd27940 ; end
      8'd77  : begin base = 24'd10333191; delta = 16'd28016 ; end
      8'd78  : begin base = 24'd10361208; delta = 16'd28092 ; end
      8'd79  : begin base = 24'd10389300; delta = 16'd28168 ; end
      8'd80  : begin base = 24'd10417468; delta = 16'd28245 ; end
      8'd81  : begin base = 24'd10445713; delta = 16'd28321 ; end
      8'd82  : begin base = 24'd10474034; delta = 16'd28398 ; end
      8'd83  : begin base = 24'd10502432; delta = 16'd28475 ; end
      8'd84  : begin base = 24'd10530907; delta = 16'd28552 ; end
      8'd85  : begin base = 24'd10559459; delta = 16'd28630 ; end
      8'd86  : begin base = 24'd10588089; delta = 16'd28707 ; end
      8'd87  : begin base = 24'd10616796; delta = 16'd28785 ; end
      8'd88  : begin base = 24'd10645581; delta = 16'd28863 ; end
      8'd89  : begin base = 24'd10674444; delta = 16'd28941 ; end
      8'd90  : begin base = 24'd10703385; delta = 16'd29020 ; end
      8'd91  : begin base = 24'd10732405; delta = 16'd29098 ; end
      8'd92  : begin base = 24'd10761504; delta = 16'd29177 ; end
      8'd93  : begin base = 24'd10790681; delta = 16'd29256 ; end
      8'd94  : begin base = 24'd10819937; delta = 16'd29336 ; end
      8'd95  : begin base = 24'd10849273; delta = 16'd29415 ; end
      8'd96  : begin base = 24'd10878689; delta = 16'd29495 ; end
      8'd97  : begin base = 24'd10908184; delta = 16'd29575 ; end
      8'd98  : begin base = 24'd10937759; delta = 16'd29655 ; end
      8'd99  : begin base = 24'd10967414; delta = 16'd29736 ; end
      8'd100 : begin base = 24'd10997150; delta = 16'd29816 ; end
      8'd101 : begin base = 24'd11026966; delta = 16'd29897 ; end
      8'd102 : begin base = 24'd11056863; delta = 16'd29978 ; end
      8'd103 : begin base = 24'd11086841; delta = 16'd30059 ; end
      8'd104 : begin base = 24'd11116901; delta = 16'd30141 ; end
      8'd105 : begin base = 24'd11147042; delta = 16'd30223 ; end
      8'd106 : begin base = 24'd11177265; delta = 16'd30305 ; end
      8'd107 : begin base = 24'd11207569; delta = 16'd30387 ; end
      8'd108 : begin base = 24'd11237956; delta = 16'd30469 ; end
      8'd109 : begin base = 24'd11268425; delta = 16'd30552 ; end
      8'd110 : begin base = 24'd11298977; delta = 16'd30635 ; end
      8'd111 : begin base = 24'd11329612; delta = 16'd30718 ; end
      8'd112 : begin base = 24'd11360329; delta = 16'd30801 ; end
      8'd113 : begin base = 24'd11391130; delta = 16'd30884 ; end
      8'd114 : begin base = 24'd11422015; delta = 16'd30968 ; end
      8'd115 : begin base = 24'd11452983; delta = 16'd31052 ; end
      8'd116 : begin base = 24'd11484035; delta = 16'd31136 ; end
      8'd117 : begin base = 24'd11515172; delta = 16'd31221 ; end
      8'd118 : begin base = 24'd11546392; delta = 16'd31305 ; end
      8'd119 : begin base = 24'd11577698; delta = 16'd31390 ; end
      8'd120 : begin base = 24'd11609088; delta = 16'd31475 ; end
      8'd121 : begin base = 24'd11640564; delta = 16'd31561 ; end
      8'd122 : begin base = 24'd11672124; delta = 16'd31646 ; end
      8'd123 : begin base = 24'd11703771; delta = 16'd31732 ; end
      8'd124 : begin base = 24'd11735503; delta = 16'd31818 ; end
      8'd125 : begin base = 24'd11767321; delta = 16'd31904 ; end
      8'd126 : begin base = 24'd11799225; delta = 16'd31991 ; end
      8'd127 : begin base = 24'd11831216; delta = 16'd32078 ; end
      8'd128 : begin base = 24'd11863294; delta = 16'd32165 ; end
      8'd129 : begin base = 24'd11895459; delta = 16'd32252 ; end
      8'd130 : begin base = 24'd11927711; delta = 16'd32339 ; end
      8'd131 : begin base = 24'd11960050; delta = 16'd32427 ; end
      8'd132 : begin base = 24'd11992477; delta = 16'd32515 ; end
      8'd133 : begin base = 24'd12024992; delta = 16'd32603 ; end
      8'd134 : begin base = 24'd12057595; delta = 16'd32691 ; end
      8'd135 : begin base = 24'd12090286; delta = 16'd32780 ; end
      8'd136 : begin base = 24'd12123066; delta = 16'd32869 ; end
      8'd137 : begin base = 24'd12155935; delta = 16'd32958 ; end
      8'd138 : begin base = 24'd12188893; delta = 16'd33047 ; end
      8'd139 : begin base = 24'd12221941; delta = 16'd33137 ; end
      8'd140 : begin base = 24'd12255078; delta = 16'd33227 ; end
      8'd141 : begin base = 24'd12288305; delta = 16'd33317 ; end
      8'd142 : begin base = 24'd12321622; delta = 16'd33407 ; end
      8'd143 : begin base = 24'd12355029; delta = 16'd33498 ; end
      8'd144 : begin base = 24'd12388527; delta = 16'd33589 ; end
      8'd145 : begin base = 24'd12422116; delta = 16'd33680 ; end
      8'd146 : begin base = 24'd12455795; delta = 16'd33771 ; end
      8'd147 : begin base = 24'd12489567; delta = 16'd33863 ; end
      8'd148 : begin base = 24'd12523429; delta = 16'd33954 ; end
      8'd149 : begin base = 24'd12557384; delta = 16'd34046 ; end
      8'd150 : begin base = 24'd12591430; delta = 16'd34139 ; end
      8'd151 : begin base = 24'd12625569; delta = 16'd34231 ; end
      8'd152 : begin base = 24'd12659800; delta = 16'd34324 ; end
      8'd153 : begin base = 24'd12694125; delta = 16'd34417 ; end
      8'd154 : begin base = 24'd12728542; delta = 16'd34511 ; end
      8'd155 : begin base = 24'd12763052; delta = 16'd34604 ; end
      8'd156 : begin base = 24'd12797657; delta = 16'd34698 ; end
      8'd157 : begin base = 24'd12832355; delta = 16'd34792 ; end
      8'd158 : begin base = 24'd12867147; delta = 16'd34886 ; end
      8'd159 : begin base = 24'd12902033; delta = 16'd34981 ; end
      8'd160 : begin base = 24'd12937014; delta = 16'd35076 ; end
      8'd161 : begin base = 24'd12972090; delta = 16'd35171 ; end
      8'd162 : begin base = 24'd13007261; delta = 16'd35266 ; end
      8'd163 : begin base = 24'd13042527; delta = 16'd35362 ; end
      8'd164 : begin base = 24'd13077889; delta = 16'd35458 ; end
      8'd165 : begin base = 24'd13113347; delta = 16'd35554 ; end
      8'd166 : begin base = 24'd13148900; delta = 16'd35650 ; end
      8'd167 : begin base = 24'd13184551; delta = 16'd35747 ; end
      8'd168 : begin base = 24'd13220298; delta = 16'd35844 ; end
      8'd169 : begin base = 24'd13256142; delta = 16'd35941 ; end
      8'd170 : begin base = 24'd13292083; delta = 16'd36038 ; end
      8'd171 : begin base = 24'd13328121; delta = 16'd36136 ; end
      8'd172 : begin base = 24'd13364257; delta = 16'd36234 ; end
      8'd173 : begin base = 24'd13400491; delta = 16'd36332 ; end
      8'd174 : begin base = 24'd13436824; delta = 16'd36431 ; end
      8'd175 : begin base = 24'd13473255; delta = 16'd36530 ; end
      8'd176 : begin base = 24'd13509784; delta = 16'd36629 ; end
      8'd177 : begin base = 24'd13546413; delta = 16'd36728 ; end
      8'd178 : begin base = 24'd13583141; delta = 16'd36828 ; end
      8'd179 : begin base = 24'd13619969; delta = 16'd36927 ; end
      8'd180 : begin base = 24'd13656896; delta = 16'd37028 ; end
      8'd181 : begin base = 24'd13693924; delta = 16'd37128 ; end
      8'd182 : begin base = 24'd13731052; delta = 16'd37229 ; end
      8'd183 : begin base = 24'd13768281; delta = 16'd37330 ; end
      8'd184 : begin base = 24'd13805610; delta = 16'd37431 ; end
      8'd185 : begin base = 24'd13843041; delta = 16'd37532 ; end
      8'd186 : begin base = 24'd13880573; delta = 16'd37634 ; end
      8'd187 : begin base = 24'd13918207; delta = 16'd37736 ; end
      8'd188 : begin base = 24'd13955943; delta = 16'd37838 ; end
      8'd189 : begin base = 24'd13993782; delta = 16'd37941 ; end
      8'd190 : begin base = 24'd14031723; delta = 16'd38044 ; end
      8'd191 : begin base = 24'd14069767; delta = 16'd38147 ; end
      8'd192 : begin base = 24'd14107914; delta = 16'd38250 ; end
      8'd193 : begin base = 24'd14146164; delta = 16'd38354 ; end
      8'd194 : begin base = 24'd14184518; delta = 16'd38458 ; end
      8'd195 : begin base = 24'd14222976; delta = 16'd38562 ; end
      8'd196 : begin base = 24'd14261539; delta = 16'd38667 ; end
      8'd197 : begin base = 24'd14300206; delta = 16'd38772 ; end
      8'd198 : begin base = 24'd14338978; delta = 16'd38877 ; end
      8'd199 : begin base = 24'd14377855; delta = 16'd38982 ; end
      8'd200 : begin base = 24'd14416837; delta = 16'd39088 ; end
      8'd201 : begin base = 24'd14455925; delta = 16'd39194 ; end
      8'd202 : begin base = 24'd14495119; delta = 16'd39300 ; end
      8'd203 : begin base = 24'd14534419; delta = 16'd39407 ; end
      8'd204 : begin base = 24'd14573826; delta = 16'd39514 ; end
      8'd205 : begin base = 24'd14613340; delta = 16'd39621 ; end
      8'd206 : begin base = 24'd14652960; delta = 16'd39728 ; end
      8'd207 : begin base = 24'd14692689; delta = 16'd39836 ; end
      8'd208 : begin base = 24'd14732524; delta = 16'd39944 ; end
      8'd209 : begin base = 24'd14772468; delta = 16'd40052 ; end
      8'd210 : begin base = 24'd14812521; delta = 16'd40161 ; end
      8'd211 : begin base = 24'd14852681; delta = 16'd40270 ; end
      8'd212 : begin base = 24'd14892951; delta = 16'd40379 ; end
      8'd213 : begin base = 24'd14933330; delta = 16'd40488 ; end
      8'd214 : begin base = 24'd14973818; delta = 16'd40598 ; end
      8'd215 : begin base = 24'd15014417; delta = 16'd40708 ; end
      8'd216 : begin base = 24'd15055125; delta = 16'd40819 ; end
      8'd217 : begin base = 24'd15095943; delta = 16'd40929 ; end
      8'd218 : begin base = 24'd15136873; delta = 16'd41040 ; end
      8'd219 : begin base = 24'd15177913; delta = 16'd41151 ; end
      8'd220 : begin base = 24'd15219064; delta = 16'd41263 ; end
      8'd221 : begin base = 24'd15260327; delta = 16'd41375 ; end
      8'd222 : begin base = 24'd15301702; delta = 16'd41487 ; end
      8'd223 : begin base = 24'd15343189; delta = 16'd41600 ; end
      8'd224 : begin base = 24'd15384789; delta = 16'd41712 ; end
      8'd225 : begin base = 24'd15426501; delta = 16'd41825 ; end
      8'd226 : begin base = 24'd15468327; delta = 16'd41939 ; end
      8'd227 : begin base = 24'd15510266; delta = 16'd42053 ; end
      8'd228 : begin base = 24'd15552318; delta = 16'd42167 ; end
      8'd229 : begin base = 24'd15594485; delta = 16'd42281 ; end
      8'd230 : begin base = 24'd15636766; delta = 16'd42396 ; end
      8'd231 : begin base = 24'd15679162; delta = 16'd42510 ; end
      8'd232 : begin base = 24'd15721672; delta = 16'd42626 ; end
      8'd233 : begin base = 24'd15764298; delta = 16'd42741 ; end
      8'd234 : begin base = 24'd15807039; delta = 16'd42857 ; end
      8'd235 : begin base = 24'd15849896; delta = 16'd42973 ; end
      8'd236 : begin base = 24'd15892870; delta = 16'd43090 ; end
      8'd237 : begin base = 24'd15935960; delta = 16'd43207 ; end
      8'd238 : begin base = 24'd15979167; delta = 16'd43324 ; end
      8'd239 : begin base = 24'd16022490; delta = 16'd43441 ; end
      8'd240 : begin base = 24'd16065932; delta = 16'd43559 ; end
      8'd241 : begin base = 24'd16109491; delta = 16'd43677 ; end
      8'd242 : begin base = 24'd16153168; delta = 16'd43796 ; end
      8'd243 : begin base = 24'd16196964; delta = 16'd43914 ; end
      8'd244 : begin base = 24'd16240878; delta = 16'd44033 ; end
      8'd245 : begin base = 24'd16284912; delta = 16'd44153 ; end
      8'd246 : begin base = 24'd16329065; delta = 16'd44273 ; end
      8'd247 : begin base = 24'd16373337; delta = 16'd44393 ; end
      8'd248 : begin base = 24'd16417730; delta = 16'd44513 ; end
      8'd249 : begin base = 24'd16462243; delta = 16'd44634 ; end
      8'd250 : begin base = 24'd16506877; delta = 16'd44755 ; end
      8'd251 : begin base = 24'd16551631; delta = 16'd44876 ; end
      8'd252 : begin base = 24'd16596507; delta = 16'd44998 ; end
      8'd253 : begin base = 24'd16641505; delta = 16'd45120 ; end
      8'd254 : begin base = 24'd16686625; delta = 16'd45242 ; end
      8'd255 : begin base = 24'd16731867; delta = 16'd45365 ; end
      default:   begin base = 24'd8388616; delta = 16'd22744; end
    endcase
  end

endmodule
