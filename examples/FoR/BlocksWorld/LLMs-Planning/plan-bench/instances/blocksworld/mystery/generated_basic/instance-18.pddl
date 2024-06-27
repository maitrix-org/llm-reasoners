

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
(craves b c)
(craves c d))
)
)


