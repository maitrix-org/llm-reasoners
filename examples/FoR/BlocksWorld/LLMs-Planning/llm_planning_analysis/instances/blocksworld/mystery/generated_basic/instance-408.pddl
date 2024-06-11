

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(craves b d)
(planet c)
(craves d a)
(province b)
)
(:goal
(and
(craves a c)
(craves b a)
(craves c d))
)
)


