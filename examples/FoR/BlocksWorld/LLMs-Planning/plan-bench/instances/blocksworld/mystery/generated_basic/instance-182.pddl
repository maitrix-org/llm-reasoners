

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(craves b a)
(planet c)
(craves d b)
(province d)
)
(:goal
(and
(craves a b)
(craves b c)
(craves d a))
)
)


