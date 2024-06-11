

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b a)
(on c d)
(on d b)
(clear c)
)
(:goal
(and
(on b d)
(on c a)
(on d c))
)
)


