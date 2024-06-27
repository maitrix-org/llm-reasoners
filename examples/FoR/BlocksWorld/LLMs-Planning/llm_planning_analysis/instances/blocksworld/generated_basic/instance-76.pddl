

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a b)
(ontable b)
(on c d)
(ontable d)
(clear a)
(clear c)
)
(:goal
(and
(on b a)
(on d c))
)
)


