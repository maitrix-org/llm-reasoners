(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l b k i)
(:init 
(harmony)
(planet l)
(planet b)
(planet k)
(planet i)
(province l)
(province b)
(province k)
(province i)
)
(:goal
(and
(craves l b)
(craves b k)
(craves k i)
)))