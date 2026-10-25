@testable import StrataCore
import XCTest

final class SkyClockTests: XCTestCase {
    // The day opens on blue hour: the sun just under the horizon, rising.
    func testOpensBeforeSunrise() {
        var clock = SkyClock()
        XCTAssertLessThan(clock.sunDirection.y, 0)
        let before = clock.sunDirection.y
        clock.advance(1)
        XCTAssertGreaterThan(clock.sunDirection.y, before)
    }
}
