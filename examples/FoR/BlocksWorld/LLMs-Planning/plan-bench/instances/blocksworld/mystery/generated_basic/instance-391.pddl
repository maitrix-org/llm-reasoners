

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b a)
(planet c)
(planet d)
(province b)
(province c)
(province d)
)
(:goal
(and
(craves c b)
(craves d a))
)
)


