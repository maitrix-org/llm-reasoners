

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b a)
(planet c)
(craves d b)
(province c)
(province d)
)
(:goal
(and
(craves a b)
(craves d c))
)
)


